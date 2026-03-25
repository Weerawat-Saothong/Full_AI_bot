# ai/predict.py
import numpy as np
import logging
from ai.model import AITradingModel
from config.config import MODEL_PATH

logger = logging.getLogger(__name__)

class SignalAnalyzer:
    def __init__(self):
        self.model = AITradingModel()
        if self.model.load(MODEL_PATH):
            logger.info("✅ SignalAnalyzer: AI Model loaded successfully")
        else:
            logger.warning("⚠️ SignalAnalyzer: Model file not found, using raw architecture")

    def analyze(self, features_df):
        """วิเคราะห์สัญญาณจาก Feature ล่าสุด"""
        try:
            # ใช้ข้อมูลแถวล่าสุด
            last_features = features_df.tail(1)
            
            # เลือกฟีเจอร์ที่ใช้ฝึก
            feature_cols = ['ema_9', 'ema_21', 'rsi_14', 'atr_14', 'macd']
            X = last_features[feature_cols].values
            
            # ทำนายความมั่นใจ (Confidence)
            confidence = self.model.predict(X)[0][0]
            
            # ตัดสินใจ Buy/Sell/Wait
            # สมมติว่า target คือ 1=Buy/Profit, 0=Sell/Loss (หรือตามที่ฝึกมา)
            # ในที่นี้ใช้เกณฑ์ความมั่นใจ > 0.55 สำหรับ Buy และ < 0.45 สำหรับ Sell
            if confidence > 0.55:
                return 'BUY', float(confidence)
            elif confidence < 0.45:
                return 'SELL', float(1 - confidence)
            else:
                return 'WAIT', float(confidence)
                
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return 'ERROR', 0.0
