import os
import ssl
import random
import numpy as np

# Automatically handle SSL verification issues on macOS for downloading pretrained weights
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

class Config:
    # Reproducibility
    SEED = 42
    
    # Image & Dataset Configurations
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
    CHANNELS = 3
    BATCH_SIZE = 32
    
    # Training Hyperparameters
    EPOCHS = 15  # Efficient & optimal for fine-tuning baseline/transfer models
    LEARNING_RATE = 1e-4
    FINE_TUNE_LEARNING_RATE = 1e-5
    CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
    
    # Paths
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "chest_xray_data", "chest_xray")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
    TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
    
    @classmethod
    def setup(cls):
        """Create necessary output directories and set random seeds."""
        for d in [cls.OUTPUT_DIR, cls.MODEL_DIR, cls.FIGURE_DIR, cls.TABLE_DIR]:
            os.makedirs(d, exist_ok=True)
            
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        try:
            import tensorflow as tf
            tf.random.set_seed(cls.SEED)
        except ImportError:
            pass

# Initialize directories and seeds upon configuration load
Config.setup()
