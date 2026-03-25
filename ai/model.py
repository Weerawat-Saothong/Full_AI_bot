# ai/model.py
# AI Model Architecture
import tensorflow as tf
from tensorflow.keras import layers, models
import os

class AITradingModel:
    def __init__(self, input_features=6):
        self.input_features = input_features
        self.model = self._build_model()
    
    def _build_model(self):
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.input_features,)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid') # 1=Profit, 0=Loss prediction
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, model_path='models/ai_trading_model.h5'):
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        return history
    
    def predict(self, features):
        return self.model.predict(features)

    def load(self, model_path):
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            return True
        return False
