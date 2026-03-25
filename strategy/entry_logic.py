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
        """ตรวจสอบว่าควรเข้าไม้หรือไม่"""
        # 1. สร้าง Feature
        features = self.engineer.generate_features(market_data_df)
        
        # 2. ให้ AI วิเคราะห์
        signal, confidence = self.analyzer.analyze(features)
        
        # 3. เพิ่มกฎความปลอดภัย (Technical Filters)
        if signal == 'BUY':
            last_row = features.iloc[-1]
            # กฎเพิ่มเติม: RSI ไม่ควร Overbought เกินไป
            if last_row['rsi_14'] > 75:
                # logger.info("🚫 Buy Signal rejected: RSI too high")
                return 'WAIT', confidence, last_row.to_dict()
                
        if signal == 'SELL':
            last_row = features.iloc[-1]
            # กฎเพิ่มเติม: RSI ไม่ควร Oversold เกิ่นไป
            if last_row['rsi_14'] < 25:
                # logger.info("🚫 Sell Signal rejected: RSI too low")
                return 'WAIT', confidence, last_row.to_dict()
        
        return signal, confidence, features.iloc[-1].to_dict()
