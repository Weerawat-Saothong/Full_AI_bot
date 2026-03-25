# data/feature_engineering.py
# สร้างฟีเจอร์
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass
    
    def generate_features(self, df):
        """สร้างฟีเจอร์จากดิบข้อมูล"""
        # Copy df to avoid original changes
        data = df.copy()
        
        # EMA
        data['ema_9'] = data['close'].ewm(span=9, adjust=False).mean()
        data['ema_21'] = data['close'].ewm(span=21, adjust=False).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi_14'] = 100 - (100 / (1 + rs))
        
        # ✅ ATR (Proper Calculation for SL/TP)
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        data['atr_14'] = true_range.rolling(14).mean()
        
        # MACD
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp1 - exp2
        
        return data.dropna()
