# data/feature_engineering.py
# สร้างฟีเจอร์
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass
    
    def generate_features(self, df):
        """สร้างฟีเจอร์จากดิบข้อมูล"""
        # EMA
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # ATR (Simple version)
        df['atr_14'] = df['high'] - df['low'] # Placeholder
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        
        return df.dropna()
