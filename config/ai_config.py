# config/ai_config.py
# ✅ AI Specific Parameters

# AI Model Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Feature list used for training/prediction
FEATURE_COLUMNS = ['ema_9', 'ema_21', 'rsi_14', 'atr_14', 'macd', 'confidence']