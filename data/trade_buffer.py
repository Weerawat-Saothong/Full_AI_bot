# data/trade_buffer.py
import pandas as pd
import os
import logging
from config.config import TRADE_BUFFER_PATH

logger = logging.getLogger(__name__)

class TradeBuffer:
    def __init__(self):
        self.csv_path = TRADE_BUFFER_PATH
        self._init_csv()

    def _init_csv(self):
        """สร้างไฟล์ CSV พร้อม Header ถ้ายังไม่มี (ใช้ utf-8-sig สำหรับ Windows)"""
        if not os.path.exists(self.csv_path):
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            df = pd.DataFrame(columns=[
                'timestamp', 'symbol', 'action', 'entry_price', 'exit_price', 
                'pnl', 'ema_9', 'ema_21', 'rsi_14', 'atr_14', 'macd', 'result'
            ])
            df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"📁 Initialized new trade buffer: {self.csv_path}")

    def log_trade_complete(self, trade_data):
        """บันทึกข้อมูลเทรดที่จบแล้วลง CSV"""
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
            
            # คำนวณ PnL และ Result
            pnl = trade_data.get('pnl', 0)
            result = 1 if pnl > 0 else 0
            
            new_row = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': trade_data.get('symbol', 'XAUUSD'),
                'action': trade_data.get('action'),
                'entry_price': trade_data.get('entry_price'),
                'exit_price': trade_data.get('exit_price'),
                'pnl': pnl,
                'ema_9': trade_data.get('ema_9'),
                'ema_21': trade_data.get('ema_21'),
                'rsi_14': trade_data.get('rsi_14'),
                'atr_14': trade_data.get('atr_14'),
                'macd': trade_data.get('macd'),
                'result': result
            }
            
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"✅ Trade logged: {trade_data.get('action')} PnL: {pnl}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log trade: {e}")

    def get_training_data(self):
        """โหลดข้อมูลสำหรับฝึก AI"""
        if not os.path.exists(self.csv_path):
            return None, None
            
        df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
        if len(df) < 10: # ขั้นต่ำสำหรับตรวจสอบ
            return None, None
            
        feature_cols = ['ema_9', 'ema_21', 'rsi_14', 'atr_14', 'macd']
        X = df[feature_cols].values
        y = df['result'].values
        
        return X, y

    def get_stats(self):
        """คำนวณสถิติเบื้องต้น"""
        if not os.path.exists(self.csv_path):
            return {"total_trades": 0, "win_rate": 0}
            
        df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
        if len(df) == 0:
            return {"total_trades": 0, "win_rate": 0}
            
        total = len(df)
        wins = len(df[df['result'] == 1])
        win_rate = wins / total
        
        return {
            "total_trades": total,
            "win_rate": win_rate,
            "total_pnl": df['pnl'].sum()
        }