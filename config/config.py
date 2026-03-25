import os
from dotenv import load_dotenv

# โหลดค่าจากไฟล์ .env
load_dotenv()

# ✅ MT5 ACCOUNT SETTINGS
MT5_LOGIN = int(os.getenv("MT5_LOGIN", 12345678))          # เลขบัญชี MT5
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "your_password") # รหัสผ่าน
MT5_SERVER = os.getenv("MT5_SERVER", "Exness-MT5-Real")   # เซิร์ฟเวอร์
MT5_SYMBOL = os.getenv("MT5_SYMBOL", "XAUUSD")            # สัญลักษณ์ที่เทรด

# ✅ TRADING PARAMETERS
LOT_SIZE = float(os.getenv("LOT_SIZE", 0.01))             # ขนาด Lot เริ่มต้น
MAX_TRADES = int(os.getenv("MAX_TRADES", 1))              # เปิดพร้อมกันสูงสุดกี่ไม้
SL_PIPS = int(os.getenv("SL_PIPS", 500))                  # Stop Loss (Points)
TP_PIPS = int(os.getenv("TP_PIPS", 1000))                 # Take Profit (Points)

# ✅ AUTO-RETRAIN SETTINGS
AUTO_RETRAIN_ENABLED = True
RETRAIN_INTERVAL_HOURS = 24
MIN_TRAINING_SAMPLES = 500
RETRAIN_ON_PERFORMANCE_DROP = True
PERFORMANCE_THRESHOLD = 0.50
MAX_MODEL_VERSIONS = 5

# Model Files
MODEL_PATH = "models/ai_trading_model.h5"
MODEL_BACKUP_DIR = "models/backups/"
MODEL_REGISTRY = "models/model_registry.json"
TRADE_BUFFER_PATH = "data/trade_buffer.csv"

# Validation
VALIDATION_BEFORE_DEPLOY = True
MIN_VALIDATION_ACCURACY = 0.55