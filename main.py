import asyncio
import aiohttp
import json
import math
import logging
import ssl
from datetime import datetime
import pandas as pd
from binance import AsyncClient
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_interval(interval_str):
    """
    Преобразует строковый таймфрейм (например, "1m", "1h", "1d") в количество секунд.
    """
    unit = interval_str[-1]
    number = int(interval_str[:-1])
    if unit == 'm':
        return number * 60
    elif unit == 'h':
        return number * 3600
    elif unit == 'd':
        return number * 86400
    else:
        raise ValueError("Unsupported interval format")


async def send_telegram_message(message, telegram_bot_token, telegram_chat_id):
    """
    Асинхронная отправка сообщения в Telegram.
    """
    url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=payload) as resp:
                if resp.status != 200:
                    logger.error(f"Ошибка отправки Telegram сообщения: {await resp.text()}")
                else:
                    logger.info("Сообщение отправлено в Telegram")
    except Exception as e:
        logger.error(f"Ошибка отправки Telegram сообщения: {e}")


async def get_symbol_step_size_async(client, symbol):
    """
    Получает stepSize для указанного символа через асинхронного клиента Binance.
    """
    info = await client.futures_exchange_info()
    for s in info["symbols"]:
        if s["symbol"] == symbol:
            for f in s["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    return float(f["stepSize"])
    return None


def round_quantity(quantity, step_size):
    """
    Округление количества согласно шагу (stepSize).
    """
    decimals = int(round(-math.log(step_size, 10), 0))
    return round(quantity, decimals)


def calculate_indicators(closed_prices, ma_period, sigma1, sigma2, sigma3):
    """
    Рассчитывает MA и уровни Bollinger Bands по последним закрытым свечам.
    """
    if len(closed_prices) < ma_period:
        return None
    series = pd.Series(closed_prices[-ma_period:])
    ma = series.mean()
    std = series.std()
    indicators = {
        "MA": ma,
        "std": std,
        "upper_band_1": ma + sigma1 * std,
        "lower_band_1": ma - sigma1 * std,
        "upper_band_2": ma + sigma2 * std,
        "lower_band_2": ma - sigma2 * std,
        "upper_band_3": ma + sigma3 * std,
        "lower_band_3": ma - sigma3 * std,
    }
    return indicators


class AsyncTradingBot:
    def __init__(self, config, client):
        self.config = config
        self.client = client
        self.symbol1 = config["symbol1"]
        self.symbol2 = config["symbol2"]
        self.candle_interval = config["candle_interval"]
        self.candle_interval_seconds = parse_interval(self.candle_interval)
        self.ma_period = config["ma_period"]
        self.sigma1 = config["first_sigma"]
        self.sigma2 = config["second_sigma"]
        self.sigma3 = config["third_sigma"]
        self.order_amount = config["order_amount"]
        self.telegram_bot_token = config["telegram_bot_token"]
        self.telegram_chat_id = config["telegram_chat_id"]

        self.last_price_symbol1 = None
        self.last_price_symbol2 = None
        self.current_candle = None  # {'start', 'open', 'high', 'low', 'close'}
        self.closed_candles = []    # список закрытых цен синтетических свечей
        self.indicators = None
        self.positions = []         # список открытых позиций
        self.signal_lock = asyncio.Lock()
        self.upper_stop_closed_flag = False
        self.lower_stop_closed_flag = False

        # Флаг активности бота (если False, сигналы не обрабатываются)
        self.active = True

        # Задачи для websocket-подключений
        self.ws_tasks = []

    def position_exists(self, direction, sigma_trigger):
        """
        Проверяет, существует ли уже открытая позиция для данного направления и уровня сигмы.
        """
        for pos in self.positions:
            if pos["type"] == direction and pos["sigma_trigger"] == sigma_trigger:
                return True
        return False

    async def load_historical_data(self):
        url = "https://fapi.binance.com/fapi/v1/klines"
        params1 = {"symbol": self.symbol1, "interval": self.candle_interval, "limit": self.ma_period}
        params2 = {"symbol": self.symbol2, "interval": self.candle_interval, "limit": self.ma_period}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params1) as resp1:
                    data1 = await resp1.json()
                async with session.get(url, params=params2) as resp2:
                    data2 = await resp2.json()
            synthetic_prices = []
            length = min(len(data1), len(data2))
            for i in range(length):
                close1 = float(data1[i][4])
                close2 = float(data2[i][4])
                if close2 != 0:
                    synthetic_prices.append(close1 / close2)
            self.closed_candles = synthetic_prices
            logger.info(f"Загружено исторических свечей: {len(self.closed_candles)}")
            self.indicators = calculate_indicators(self.closed_candles, self.ma_period, self.sigma1, self.sigma2, self.sigma3)
            if self.indicators:
                logger.info(
                    f"Индикаторы (исторические данные): MA={self.indicators['MA']:.5f}, "
                    f"+1 сигма={self.indicators['upper_band_1']:.5f}, -1 сигма={self.indicators['lower_band_1']:.5f}, "
                    f"+2 сигма={self.indicators['upper_band_2']:.5f}, -2 сигма={self.indicators['lower_band_2']:.5f}, "
                    f"+3 сигма={self.indicators['upper_band_3']:.5f}, -3 сигма={self.indicators['lower_band_3']:.5f}"
                )
            else:
                logger.info("Не удалось вычислить индикаторы из исторических данных")
        except Exception as e:
            logger.error(f"Ошибка загрузки исторических данных: {e}")

    async def update_price(self, symbol, price):
        now = datetime.utcnow()
        if symbol == self.symbol1:
            self.last_price_symbol1 = price
        elif symbol == self.symbol2:
            self.last_price_symbol2 = price

        if (self.last_price_symbol1 is not None and
            self.last_price_symbol2 is not None and
            self.last_price_symbol2 != 0):
            synthetic_price = self.last_price_symbol1 / self.last_price_symbol2
            logger.info(f"Синтетическая цена: {synthetic_price:.5f}")

            candle_start = datetime.utcfromtimestamp(
                (now.timestamp() // self.candle_interval_seconds) * self.candle_interval_seconds
            )
            if self.current_candle is None:
                self.current_candle = {'start': candle_start, 'open': synthetic_price, 'high': synthetic_price,
                                        'low': synthetic_price, 'close': synthetic_price}
            elif self.current_candle['start'] != candle_start:
                closed_price = self.current_candle['close']
                self.closed_candles.append(closed_price)
                logger.info(f"Свеча закрыта. Начало: {self.current_candle['start'].strftime('%H:%M:%S')}, Закрытие: {closed_price:.5f}")
                self.indicators = calculate_indicators(self.closed_candles, self.ma_period, self.sigma1, self.sigma2, self.sigma3)
                if self.indicators:
                    logger.info(
                        f"Индикаторы: MA={self.indicators['MA']:.5f}, "
                        f"+1 сигма={self.indicators['upper_band_1']:.5f}, -1 сигма={self.indicators['lower_band_1']:.5f}, "
                        f"+2 сигма={self.indicators['upper_band_2']:.5f}, -2 сигма={self.indicators['lower_band_2']:.5f}, "
                        f"+3 сигма={self.indicators['upper_band_3']:.5f}, -3 сигма={self.indicators['lower_band_3']:.5f}"
                    )
                else:
                    logger.info("Недостаточно данных для расчёта индикаторов.")
                self.current_candle = {'start': candle_start, 'open': synthetic_price, 'high': synthetic_price,
                                        'low': synthetic_price, 'close': synthetic_price}
            else:
                self.current_candle['high'] = max(self.current_candle['high'], synthetic_price)
                self.current_candle['low'] = min(self.current_candle['low'], synthetic_price)
                self.current_candle['close'] = synthetic_price

            if self.active:
                await self.check_live_signals(synthetic_price)

    async def send_market_order(self, symbol, side, quantity=None):
        try:
            if quantity is None:
                price = self.last_price_symbol1 if symbol == self.symbol1 else self.last_price_symbol2
                if price is None or price <= 0:
                    logger.warning(f"Нет валидной цены для {symbol} при отправке ордера.")
                    return
                step_size = await get_symbol_step_size_async(self.client, symbol)
                if step_size is None:
                    logger.warning(f"Не удалось получить stepSize для {symbol}.")
                    return
                quantity = self.order_amount / price
                quantity = round_quantity(quantity, step_size)
            logger.info(f"Отправка ордера: {side} {symbol}, количество: {quantity:.5f} (используемая цена: {self.last_price_symbol1 if symbol == self.symbol1 else self.last_price_symbol2:.5f})")
            order = await self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity
            )
            logger.info(f"Ордер выполнен для {symbol}: {order}")
        except Exception as e:
            logger.error(f"Ошибка при отправке ордера для {symbol}: {e}")

    async def open_position(self, direction, price, sigma_trigger):
        if self.position_exists(direction, sigma_trigger):
            logger.info(f"Позиция для {direction} с sigma_trigger={sigma_trigger} уже открыта.")
            return

        if self.last_price_symbol1 is None or self.last_price_symbol2 is None:
            logger.warning("Нет доступных цен для открытия позиции.")
            return

        step_size1 = await get_symbol_step_size_async(self.client, self.symbol1)
        quantity1 = round_quantity(self.order_amount / self.last_price_symbol1, step_size1)

        step_size2 = await get_symbol_step_size_async(self.client, self.symbol2)
        quantity2 = round_quantity(self.order_amount / self.last_price_symbol2, step_size2)

        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        if direction == "upper":
            msg = (f"[Сигнал] {timestamp}: Цена достигла верхней {sigma_trigger} сигмы ({price:.5f}).\n"
                   f"Открытие короткой позиции: ПРОДАЁМ {self.symbol1} на ${self.order_amount} и ПОКУПАЕМ {self.symbol2} на ${self.order_amount}.")
            logger.info(msg)
            await send_telegram_message(msg, self.telegram_bot_token, self.telegram_chat_id)
            await self.send_market_order(self.symbol1, "SELL", quantity1)
            await self.send_market_order(self.symbol2, "BUY", quantity2)
        elif direction == "lower":
            msg = (f"[Сигнал] {timestamp}: Цена достигла нижней {sigma_trigger} сигмы ({price:.5f}).\n"
                   f"Открытие длинной позиции: ПОКУПАЕМ {self.symbol1} на ${self.order_amount} и ПРОДАЁМ {self.symbol2} на ${self.order_amount}.")
            logger.info(msg)
            await send_telegram_message(msg, self.telegram_bot_token, self.telegram_chat_id)
            await self.send_market_order(self.symbol1, "BUY", quantity1)
            await self.send_market_order(self.symbol2, "SELL", quantity2)
        self.positions.append({
            "type": direction,
            "entry": price,
            "time": datetime.utcnow(),
            "sigma_trigger": sigma_trigger,
            "symbol1_quantity": quantity1,
            "symbol2_quantity": quantity2
        })

    async def close_position(self, pos, price):
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        closed_by_stop = False
        if pos["type"] == "upper":
            profit = pos["entry"] - price
            msg = (f"[Тейкпрофит/Стоплосс] {timestamp}: Цена = {price:.5f}.\n")
            if price >= self.indicators["upper_band_3"]:
                msg += f"Закрытие короткой позиции по стопу. Прибыль: {profit:.5f}"
                closed_by_stop = True
            else:
                msg += f"Закрытие короткой позиции по тейкпрофиту. Прибыль: {profit:.5f}"
            logger.info(msg)
            await send_telegram_message(msg, self.telegram_bot_token, self.telegram_chat_id)
            await self.send_market_order(self.symbol1, "BUY", pos["symbol1_quantity"])
            await self.send_market_order(self.symbol2, "SELL", pos["symbol2_quantity"])
            if closed_by_stop:
                self.upper_stop_closed_flag = True
            else:
                self.upper_stop_closed_flag = False
        elif pos["type"] == "lower":
            profit = price - pos["entry"]
            msg = (f"[Тейкпрофит/Стоплосс] {timestamp}: Цена = {price:.5f}.\n")
            if price <= self.indicators["lower_band_3"]:
                msg += f"Закрытие длинной позиции по стопу. Прибыль: {profit:.5f}"
                closed_by_stop = True
            else:
                msg += f"Закрытие длинной позиции по тейкпрофиту. Прибыль: {profit:.5f}"
            logger.info(msg)
            await send_telegram_message(msg, self.telegram_bot_token, self.telegram_chat_id)
            await self.send_market_order(self.symbol1, "SELL", pos["symbol1_quantity"])
            await self.send_market_order(self.symbol2, "BUY", pos["symbol2_quantity"])
            if closed_by_stop:
                self.lower_stop_closed_flag = True
            else:
                self.lower_stop_closed_flag = False
        self.positions.remove(pos)

    async def check_live_signals(self, current_price):
        async with self.signal_lock:
            if self.indicators is None:
                return

            ma = self.indicators["MA"]

            # Проверка условий закрытия открытых позиций
            for pos in self.positions.copy():
                if pos["type"] == "upper":
                    if current_price <= ma or current_price >= self.indicators["upper_band_3"]:
                        await self.close_position(pos, current_price)
                elif pos["type"] == "lower":
                    if current_price >= ma or current_price <= self.indicators["lower_band_3"]:
                        await self.close_position(pos, current_price)

            # Обработка флага стопа для верхнего направления
            if self.upper_stop_closed_flag:
                if current_price < self.indicators["upper_band_2"]:
                    logger.info("Обратное пересечение для верхней стороны обнаружено, флаг стопа сброшен.")
                    self.upper_stop_closed_flag = False

            if not self.upper_stop_closed_flag:
                if current_price >= self.indicators["upper_band_2"] and not self.position_exists("upper", 2):
                    await self.open_position("upper", current_price, sigma_trigger=2)
                elif current_price >= self.indicators["upper_band_1"] and not self.position_exists("upper", 1):
                    await self.open_position("upper", current_price, sigma_trigger=1)

            # Обработка флага стопа для нижнего направления
            if self.lower_stop_closed_flag:
                if current_price > self.indicators["lower_band_2"]:
                    logger.info("Обратное пересечение для нижней стороны обнаружено, флаг стопа сброшен.")
                    self.lower_stop_closed_flag = False

            if not self.lower_stop_closed_flag:
                if current_price <= self.indicators["lower_band_2"] and not self.position_exists("lower", 2):
                    await self.open_position("lower", current_price, sigma_trigger=2)
                elif current_price <= self.indicators["lower_band_1"] and not self.position_exists("lower", 1):
                    await self.open_position("lower", current_price, sigma_trigger=1)

    async def start_ws(self):
        """
        Запускает websocket-подключения для обоих символов.
        """
        self.ws_tasks = [
            asyncio.create_task(listen_trades(self.symbol1, self)),
            asyncio.create_task(listen_trades(self.symbol2, self))
        ]
        logger.info(f"Websocket-подключения запущены для {self.symbol1} и {self.symbol2}.")

    async def stop_ws(self):
        """
        Останавливает websocket-подключения.
        """
        for task in self.ws_tasks:
            task.cancel()
        self.ws_tasks = []
        logger.info("Websocket-подключения остановлены.")


async def listen_trades(symbol, bot):
    stream_name = symbol.lower() + "@aggTrade"
    url = "wss://fstream.binance.com/ws"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                subscribe_msg = {"method": "SUBSCRIBE", "params": [stream_name], "id": 1}
                await ws.send_json(subscribe_msg)
                logger.info(f"Подписка на {stream_name} для {symbol} выполнена")
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = msg.json()
                        if "p" in data:
                            price = float(data["p"])
                            await bot.update_price(symbol, price)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"Ошибка в websocket для {symbol}: {msg.data}")
                        break
    except Exception as e:
        logger.error(f"Ошибка подключения к websocket для {symbol}: {e}")


# Определяем состояния для диалогов с использованием FSM
class SetupStates(StatesGroup):
    set_pair = State()
    set_volume = State()
    set_interval = State()
    set_ma_period = State()
    set_open_direction = State()


async def start_telegram_bot(trading_bot: AsyncTradingBot):
    bot = Bot(token=trading_bot.telegram_bot_token)
    storage = MemoryStorage()
    dp = Dispatcher(bot=bot, storage=storage)

    @dp.message(Command("menu"))
    async def cmd_menu(message: types.Message):
        keyboard = ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="/startbot"), KeyboardButton(text="/stopbot")],
                [KeyboardButton(text="/setpair"), KeyboardButton(text="/setvolume")],
                [KeyboardButton(text="/setinterval"), KeyboardButton(text="/setmaperiod")],
                [KeyboardButton(text="/open"), KeyboardButton(text="/closeall")],
                [KeyboardButton(text="/reloaddata"), KeyboardButton(text="/status")]
            ],
            resize_keyboard=True
        )
        await message.answer("Выберите команду:", reply_markup=keyboard)

    @dp.message(Command("startbot"))
    async def cmd_startbot(message: types.Message):
        trading_bot.active = True
        await message.reply("Торговый бот запущен.")

    @dp.message(Command("stopbot"))
    async def cmd_stopbot(message: types.Message):
        trading_bot.active = False
        await message.reply("Торговый бот остановлен.")

    # Команда /setpair с использованием FSM
    @dp.message(Command("setpair"))
    async def cmd_setpair(message: types.Message, state: FSMContext):
        await message.reply("Введите торговые пары в формате: SYMBOL1 SYMBOL2")
        await state.set_state(SetupStates.set_pair)

    @dp.message(SetupStates.set_pair)
    async def process_setpair(message: types.Message, state: FSMContext):
        try:
            sym1, sym2 = message.text.split()
            await trading_bot.stop_ws()
            trading_bot.symbol1 = sym1.upper()
            trading_bot.symbol2 = sym2.upper()
            await trading_bot.start_ws()
            await message.reply(f"Пары обновлены: {trading_bot.symbol1}, {trading_bot.symbol2}")
        except Exception as e:
            await message.reply("Ошибка в формате ввода. Используйте формат: SYMBOL1 SYMBOL2")
        await state.clear()

    # Команда /setvolume с использованием FSM
    @dp.message(Command("setvolume"))
    async def cmd_setvolume(message: types.Message, state: FSMContext):
        await message.reply("Введите объем сделки (например, 1000)")
        await state.set_state(SetupStates.set_volume)

    @dp.message(SetupStates.set_volume)
    async def process_setvolume(message: types.Message, state: FSMContext):
        try:
            volume = float(message.text)
            trading_bot.order_amount = volume
            await message.reply(f"Объем сделки обновлен: {volume}")
        except Exception as e:
            await message.reply("Ошибка в формате ввода. Объем должен быть числом.")
        await state.clear()

    # Команда /setinterval с использованием FSM
    @dp.message(Command("setinterval"))
    async def cmd_setinterval(message: types.Message, state: FSMContext):
        await message.reply("Введите таймфрейм (например, 1m, 1h, 1d)")
        await state.set_state(SetupStates.set_interval)

    @dp.message(SetupStates.set_interval)
    async def process_setinterval(message: types.Message, state: FSMContext):
        try:
            interval = message.text.strip()
            seconds = parse_interval(interval)
            trading_bot.candle_interval = interval
            trading_bot.candle_interval_seconds = seconds
            await message.reply(f"Таймфрейм обновлен: {interval}")
        except Exception as e:
            await message.reply("Ошибка в формате ввода. Используйте формат, например, 1m, 1h, 1d.")
        await state.clear()

    # Команда /setmaperiod с использованием FSM
    @dp.message(Command("setmaperiod"))
    async def cmd_setmaperiod(message: types.Message, state: FSMContext):
        await message.reply("Введите период для индикаторов MA (целое число)")
        await state.set_state(SetupStates.set_ma_period)

    @dp.message(SetupStates.set_ma_period)
    async def process_setmaperiod(message: types.Message, state: FSMContext):
        try:
            period = int(message.text)
            trading_bot.ma_period = period
            await message.reply(f"Период для индикаторов обновлен: {period}")
        except Exception as e:
            await message.reply("Ошибка в формате ввода. Введите целое число.")
        await state.clear()

    # Команда /open с использованием FSM (указание направления сделки)
    @dp.message(Command("open"))
    async def cmd_open(message: types.Message, state: FSMContext):
        await message.reply("Введите направление для открытия сделки (upper или lower)")
        await state.set_state(SetupStates.set_open_direction)

    @dp.message(SetupStates.set_open_direction)
    async def process_set_open_direction(message: types.Message, state: FSMContext):
        direction = message.text.strip().lower()
        if direction not in ['upper', 'lower']:
            await message.reply("Неверное направление. Используйте 'upper' или 'lower'.")
        else:
            if trading_bot.current_candle is None:
                await message.reply("Нет текущей свечи для определения цены.")
            else:
                current_price = trading_bot.current_candle['close']
                await trading_bot.open_position(direction, current_price, sigma_trigger=1)
                await message.reply(f"Открыта позиция {direction} по цене {current_price:.5f}")
        await state.clear()

    @dp.message(Command("closeall"))
    async def cmd_closeall(message: types.Message):
        if trading_bot.current_candle is None:
            await message.reply("Нет текущей свечи для определения цены.")
            return
        current_price = trading_bot.current_candle['close']
        if not trading_bot.positions:
            await message.reply("Нет открытых позиций для закрытия.")
            return
        positions_to_close = trading_bot.positions.copy()
        for pos in positions_to_close:
            await trading_bot.close_position(pos, current_price)
        await message.reply("Все позиции закрыты.")

    @dp.message(Command("reloaddata"))
    async def cmd_reloaddata(message: types.Message):
        await trading_bot.load_historical_data()
        await message.reply("Исторические данные успешно перезагружены.")

    @dp.message(Command("status"))
    async def cmd_status(message: types.Message):
        status_msg = f"Торговый бот {'активен' if trading_bot.active else 'неактивен'}.\n"
        status_msg += f"Пары: {trading_bot.symbol1}, {trading_bot.symbol2}\n"
        status_msg += f"Таймфрейм: {trading_bot.candle_interval}\n"
        status_msg += f"Период индикаторов: {trading_bot.ma_period}\n"
        status_msg += f"Объем сделки: {trading_bot.order_amount}\n"
        status_msg += f"Открытых позиций: {len(trading_bot.positions)}"
        await message.reply(status_msg)

    await dp.start_polling(bot)


async def main():
    # Загрузка конфигурации из файла config.json
    with open("config.json", "r") as f:
        config = json.load(f)

    ssl_context = ssl.create_default_context()
    client = await AsyncClient.create(
        api_key=config["api_key"],
        api_secret=config["api_secret"],
        tld='com',
        requests_params={'ssl': ssl_context}
    )

    trading_bot = AsyncTradingBot(config, client)
    await trading_bot.load_historical_data()

    # Запуск websocket-подключений
    await trading_bot.start_ws()

    # Запуск Telegram-бота для управления
    telegram_task = asyncio.create_task(start_telegram_bot(trading_bot))

    try:
        await telegram_task
    except Exception as e:
        logger.error(f"Ошибка в основном цикле: {e}")
    finally:
        await trading_bot.stop_ws()
        await client.close_connection()
        logger.info("Клиентская сессия Binance закрыта")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
