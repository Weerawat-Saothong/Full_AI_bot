# execution/mt5_connector.py
import MetaTrader5 as mt5
import logging
from config.config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_SYMBOL

logger = logging.getLogger(__name__)

class MT5Connector:
    def __init__(self):
        self.symbol = MT5_SYMBOL
        
    def connect(self):
        if not mt5.initialize():
            logger.error(f"❌ initialize() failed, error code = {mt5.last_error()}")
            return False
            
        authorized = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
        if authorized:
            logger.info(f"✅ Connected to MT5 Account: {MT5_LOGIN}")
            return True
        else:
            logger.error(f"❌ Failed to connect to account {MT5_LOGIN}, error code = {mt5.last_error()}")
            return False

    def get_current_price(self):
        tick = mt5.symbol_info_tick(self.symbol)
        if tick:
            return tick.bid, tick.ask
        return None, None

    def send_order(self, action, lot, sl_price=None, tp_price=None):
        """ส่งออเดอร์ Buy/Sell พร้อม SL/TP ที่คำนวณมาแล้ว"""
        price_bid, price_ask = self.get_current_price()
        if not price_bid: return None

        order_type = mt5.ORDER_TYPE_BUY if action == 'BUY' else mt5.ORDER_TYPE_SELL
        price = price_ask if action == 'BUY' else price_bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": float(sl_price) if sl_price else 0.0,
            "tp": float(tp_price) if tp_price else 0.0,
            "magic": 123456,
            "comment": "Bot_Gold_AI_Full",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC, # หรือ ORDER_FILLING_FOK ตามค่ายโบรก
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"❌ Order failed: {result.comment}")
            return None
            
        logger.info(f"🚀 Order placed: {action} at {price} | Ticket: {result.deal}")
        return result
