# main.py
# =========================
# MAIN - AI TRADING BOT 100% (Windows Optimized)
# =========================
import logging
import time
import os
import pandas as pd
from datetime import datetime, timedelta
from ai.auto_retrain import AutoRetrainer
from data.trade_buffer import TradeBuffer
from execution.mt5_connector import MT5Connector
from strategy.entry_logic import EntryStrategy
from config.config import (
    MT5_SYMBOL, LOT_SIZE, MAX_TRADES, SL_PIPS, TP_PIPS
)

# ✅ สร้างโฟลเดอร์ที่จำเป็นอัตโนมัติ (กัน Error บน Windows)
for folder in ['models', 'data', 'logs', 'models/backups']:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"📁 Created folder: {folder}")

# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/bot_trading.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GoldAIBot:
    def __init__(self):
        self.mt5 = MT5Connector()
        self.strategy = EntryStrategy()
        self.auto_retrainer = AutoRetrainer()
        self.trade_buffer = TradeBuffer()
        self.is_running = False
        
    def get_market_data(self, count=1000):
        """ดึงข้อมูลราคาล่าสุดจาก MT5"""
        import MetaTrader5 as mt5
        rates = mt5.copy_rates_from_pos(MT5_SYMBOL, mt5.TIMEFRAME_M15, 0, count)
        if rates is None: return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def run(self):
        logger.info("🚀 Gold AI Bot Starting (Windows Safe Mode)")
        
        if not self.mt5.connect():
            logger.error("❌ MT5 Connection Failed. Open MT5 and enable 'Algo Trading'.")
            return

        self.is_running = True
        last_retrain_check = datetime.now()
        
        try:
            while self.is_running:
                # 1. Check Auto-Retrain (ทุก 1 ชม.)
                if datetime.now() - last_retrain_check >= timedelta(hours=1):
                    last_retrain_check = datetime.now()
                    if self.auto_retrainer.should_retrain():
                        logger.info("🧠 Performance trigger active. Retraining...")
                        self.auto_retrainer.retrain()

                # 2. Get Market Data
                df = self.get_market_data()
                if df is None:
                    time.sleep(10)
                    continue
                
                # 3. Check for Entry (AI คำนวณสัญญาณ และ SL/TP ให้เอง)
                signal, confidence, last_features, sl, tp = self.strategy.check_entry_signal(df)
                
                if signal in ['BUY', 'SELL']:
                    import MetaTrader5 as mt5
                    positions = mt5.positions_get(symbol=MT5_SYMBOL)
                    
                    if len(positions) < MAX_TRADES:
                        logger.info(f"🎯 Signal: {signal} | Confidence: {confidence:.2%}")
                        logger.info(f"📏 AI Calculated SL: {sl:.2f} | TP: {tp:.2f}")
                        res = self.mt5.send_order(signal, LOT_SIZE, sl, tp)
                        # ในระบบสมบูรณ์ MT5 จะแจ้งปิดไม้เอง และเราต้องดึงประวัติมาบันทึก log_trade_complete
                
                time.sleep(60) # ตรวจสอบทุก 1 นาที
                
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user.")
        finally:
            import MetaTrader5 as mt5
            mt5.shutdown()
            logger.info("👋 MT5 Shutdown. Goodbye!")

if __name__ == "__main__":
    bot = GoldAIBot()
    bot.run()