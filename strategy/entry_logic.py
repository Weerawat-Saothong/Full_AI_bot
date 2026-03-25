# strategy/entry_logic.py
import logging
from ai.predict import SignalAnalyzer
from data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class EntryStrategy:
    def __init__(self):
        self.analyzer = SignalAnalyzer()
        self.engineer = FeatureEngineer()
        
    def check_entry_signal(self, market_data_df):
        """ตรวจสอบว่าควรเข้าไม้หรือไม่ พร้อมคำนวณ SL/TP อัตโนมัติ"""
        # 1. สร้าง Feature
        features = self.engineer.generate_features(market_data_df)
        last_row = features.iloc[-1]
        atr = last_row['atr_14']
        current_price = last_row['close']
        
        # 2. ให้ AI วิเคราะห์
        signal, confidence = self.analyzer.analyze(features)
        
        # ⚠️ ค่าตัวคูณสำหรับ SL/TP (ปรับจูนตามความเหมาะสม)
        SL_MULT = 2.0  # SL = 2 x ATR
        RR_RATIO = 1.5 # TP = 1.5 x SL (Risk:Reward)
        
        sl = 0.0
        tp = 0.0
        
        # 3. ตรวจสอบสัญญาณและคำนวณ SL/TP
        if signal == 'BUY':
            # กฎเพิ่มเติม: RSI ไม่ควร Overbought เกินไป
            if last_row['rsi_14'] > 75:
                return 'WAIT', confidence, last_row.to_dict(), 0, 0
            
            sl = current_price - (atr * SL_MULT)
            tp = current_price + (atr * SL_MULT * RR_RATIO)
                
        elif signal == 'SELL':
            # กฎเพิ่มเติม: RSI ไม่ควร Oversold เกิ่นไป
            if last_row['rsi_14'] < 25:
                return 'WAIT', confidence, last_row.to_dict(), 0, 0
            
            sl = current_price + (atr * SL_MULT)
            tp = current_price - (atr * SL_MULT * RR_RATIO)
        
        return signal, confidence, last_row.to_dict(), sl, tp
