# ai/auto_retrain.py
# =========================
# AUTO-RETRAIN SYSTEM
# =========================
import os
import json
import logging
import shutil
from datetime import datetime, timedelta
from ai.model import AITradingModel
from data.trade_buffer import TradeBuffer
from data.feature_engineering import FeatureEngineer
from config.config import (
    AUTO_RETRAIN_ENABLED, RETRAIN_INTERVAL_HOURS, MIN_TRAINING_SAMPLES,
    RETRAIN_ON_PERFORMANCE_DROP, PERFORMANCE_THRESHOLD, MAX_MODEL_VERSIONS,
    MODEL_PATH, MODEL_BACKUP_DIR, MODEL_REGISTRY, TRADE_BUFFER_PATH,
    VALIDATION_BEFORE_DEPLOY, MIN_VALIDATION_ACCURACY
)
from sklearn.model_selection import train_test_split
import numpy as np

logger = logging.getLogger(__name__)

class AutoRetrainer:
    def __init__(self):
        self.trade_buffer = TradeBuffer(TRADE_BUFFER_PATH)
        self.feature_engineer = FeatureEngineer()
        self.last_retrain = self._load_last_retrain_time()
        self.model_version = self._load_model_version()
        
        # สร้างโฟลเดอร์ backup
        os.makedirs(MODEL_BACKUP_DIR, exist_ok=True)
        
        logger.info(f"✅ AutoRetrainer Initialized | Version: {self.model_version}")
    
    def _load_last_retrain_time(self):
        """โหลดเวลาฝึกครั้งล่าสุด"""
        try:
            if os.path.exists(MODEL_REGISTRY):
                with open(MODEL_REGISTRY, 'r') as f:
                    registry = json.load(f)
                    last_str = registry.get('last_retrain')
                    if last_str:
                        last = datetime.fromisoformat(last_str)
                        logger.info(f"📂 Last retrain: {last}")
                        return last
        except Exception as e:
            logger.warning(f"⚠️ Could not load last retrain time: {e}")
        
        return datetime.now() - timedelta(hours=RETRAIN_INTERVAL_HOURS)
    
    def _load_model_version(self):
        """โหลด version โมเดลปัจจุบัน"""
        try:
            if os.path.exists(MODEL_REGISTRY):
                with open(MODEL_REGISTRY, 'r') as f:
                    registry = json.load(f)
                    return registry.get('version', 1)
        except:
            pass
        return 1
    
    def _save_registry(self, new_version):
        """บันทึก registry"""
        registry = {
            'version': new_version,
            'last_retrain': datetime.now().isoformat(),
            'model_path': MODEL_PATH,
            'total_retrains': new_version
        }
        
        with open(MODEL_REGISTRY, 'w') as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"💾 Registry saved: Version {new_version}")
    
    def _backup_model(self):
        """Backup โมเดลเก่าก่อนฝึกใหม่"""
        if os.path.exists(MODEL_PATH):
            backup_path = os.path.join(
                MODEL_BACKUP_DIR, 
                f"ai_trading_model_v{self.model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            )
            
            shutil.copy(MODEL_PATH, backup_path)
            logger.info(f"💾 Model backed up: {backup_path}")
            
            # ลบ backup เก่าเกิน MAX_MODEL_VERSIONS
            self._cleanup_old_backups()
    
    def _cleanup_old_backups(self):
        """ลบ backup เก่า"""
        try:
            backups = sorted([
                os.path.join(MODEL_BACKUP_DIR, f) 
                for f in os.listdir(MODEL_BACKUP_DIR) 
                if f.endswith('.h5')
            ])
            
            while len(backups) > MAX_MODEL_VERSIONS:
                os.remove(backups.pop(0))
                logger.info("🧹 Cleaned up old backup")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup failed: {e}")
    
    def should_retrain(self):
        """ตรวจสอบว่าควรฝึกใหม่หรือไม่"""
        if not AUTO_RETRAIN_ENABLED:
            logger.info("⏸️ Auto-retrain disabled")
            return False
        
        # 1. ตรวจสอบเวลา
        time_since_last = datetime.now() - self.last_retrain
        if time_since_last >= timedelta(hours=RETRAIN_INTERVAL_HOURS):
            logger.info(f"⏰ Retrain triggered by time ({time_since_last.total_seconds()/3600:.1f}h)")
            return True
        
        # 2. ตรวจสอบประสิทธิภาพ
        if RETRAIN_ON_PERFORMANCE_DROP:
            stats = self.trade_buffer.get_performance_stats(last_n_trades=50)
            
            if stats and stats['win_rate'] < PERFORMANCE_THRESHOLD:
                logger.warning(f"📉 Retrain triggered by performance (Win Rate: {stats['win_rate']:.2%})")
                return True
        
        return False
    
    def retrain(self):
        """ฝึกโมเดลใหม่"""
        logger.info("🧠 Starting Auto-Retrain...")
        
        # 1. ดึงข้อมูล
        data = self.trade_buffer.get_training_data(min_samples=MIN_TRAINING_SAMPLES)
        
        if data is None:
            logger.warning("⚠️ Insufficient data for retrain")
            return False
        
        X, y, feature_columns = data
        
        # 2. Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"📈 Train: {len(X_train)}, Validation: {len(X_val)}")
        
        # 3. Backup โมเดลเก่า
        self._backup_model()
        
        # 4. ฝึกโมเดลใหม่
        model = AITradingModel(input_features=len(feature_columns))
        
        history = model.train(
            X_train, y_train, X_val, y_val,
            epochs=50,
            batch_size=32,
            model_path=MODEL_PATH + ".tmp"  # ฝึกเป็นไฟล์ชั่วคราวก่อน
        )
        
        # 5. Validate ก่อน deploy
        if VALIDATION_BEFORE_DEPLOY:
            # สมมติว่า model.model คือ Keras model
            try:
                val_loss, val_acc = model.model.evaluate(X_val, y_val, verbose=0)
                logger.info(f"📊 Validation: Accuracy={val_acc:.4f}, Loss={val_loss:.4f}")
                
                if val_acc < MIN_VALIDATION_ACCURACY:
                    logger.warning(f"⚠️ New model accuracy ({val_acc:.2%}) below threshold ({MIN_VALIDATION_ACCURACY:.2%})")
                    logger.warning("⚠️ Keeping old model")
                    if os.path.exists(MODEL_PATH + ".tmp"):
                        os.remove(MODEL_PATH + ".tmp")
                    return False
            except Exception as e:
                logger.error(f"❌ Validation failed: {e}")
                return False
        
        # 6. Deploy โมเดลใหม่
        if os.path.exists(MODEL_PATH + ".tmp"):
            os.replace(MODEL_PATH + ".tmp", MODEL_PATH)
        
        # 7. อัพเดท registry
        new_version = self.model_version + 1
        self._save_registry(new_version)
        self.model_version = new_version
        self.last_retrain = datetime.now()
        
        logger.info(f"✅ Auto-Retrain Complete! Version: {new_version}")
        
        # 8. แจ้งเตือน (ถ้ามี Telegram/Line)
        # self._notify_retrain_complete(val_acc)
        
        return True
    
    def _notify_retrain_complete(self, accuracy):
        """แจ้งเตือนเมื่อฝึกเสร็จ"""
        message = f"""
🤖 [AUTO-RETRAIN COMPLETE]
──────────────────
Version: {self.model_version}
Accuracy: {accuracy:.2%}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
──────────────────
"""
        logger.info(message)
    
    def get_status(self):
        """ดึงสถานะ Auto-Retrain"""
        stats = self.trade_buffer.get_performance_stats()
        time_since_last = datetime.now() - self.last_retrain
        
        training_data = self.trade_buffer.get_training_data(min_samples=0)
        total_trades = len(training_data[0]) if training_data else 0
        
        return {
            'enabled': AUTO_RETRAIN_ENABLED,
            'model_version': self.model_version,
            'last_retrain': self.last_retrain.isoformat(),
            'next_retrain_in_hours': RETRAIN_INTERVAL_HOURS - (time_since_last.total_seconds() / 3600),
            'total_trades_in_buffer': total_trades,
            'performance': stats
        }
