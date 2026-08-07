import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import Config
from .models import build_baseline_cnn, build_transfer_model, get_preprocess_fn

class ModelTrainer:
    def __init__(self, model_name, img_size=Config.IMG_SIZE):
        self.model_name = model_name
        self.img_size = img_size
        self.preprocess_fn = get_preprocess_fn(model_name)
        
    def build_model(self, learning_rate=Config.LEARNING_RATE):
        if self.model_name == "BaselineCNN":
            model = build_baseline_cnn(input_shape=self.img_size + (Config.CHANNELS,))
            base_model = None
        else:
            model, base_model = build_transfer_model(
                self.model_name,
                input_shape=self.img_size + (Config.CHANNELS,),
                learning_rate=learning_rate
            )
            
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc"), keras.metrics.Recall(name="recall")]
        )
        return model, base_model

    def train_model(self, train_ds, val_ds, class_weights=None, epochs=Config.EPOCHS, fine_tune_epochs=5):
        checkpoint_path = os.path.join(Config.MODEL_DIR, f"best_{self.model_name}.keras")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint for {self.model_name} from {checkpoint_path}")
            model = keras.models.load_model(checkpoint_path)
            return model, None, None

        model, base_model = self.build_model()
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
        ]
        
        print(f"Training {self.model_name} (Phase 1)...")
        history_phase1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        if base_model is not None and fine_tune_epochs > 0:
            print(f"Fine-tuning {self.model_name} (Phase 2)...")
            base_model.trainable = True
            
            fine_tune_at = max(0, len(base_model.layers) - 30)
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
                
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=Config.FINE_TUNE_LEARNING_RATE),
                loss="binary_crossentropy",
                metrics=["accuracy", keras.metrics.AUC(name="auc"), keras.metrics.Recall(name="recall")]
            )
            
            history_phase2 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs + fine_tune_epochs,
                initial_epoch=history_phase1.epoch[-1] + 1,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )
            return model, history_phase1, history_phase2
            
        return model, history_phase1, None

    def evaluate_predictions(self, model, test_ds):
        y_true = []
        y_pred_probs = []
        
        for images, labels in test_ds:
            preds = model.predict(images, verbose=0)
            y_true.extend(labels.numpy().flatten().tolist())
            y_pred_probs.extend(preds.flatten().tolist())
            
        return np.array(y_true), np.array(y_pred_probs)
