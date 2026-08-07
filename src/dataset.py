import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras

from .config import Config

class DatasetManager:
    def __init__(self, data_dir=Config.DATA_DIR, img_size=Config.IMG_SIZE, batch_size=Config.BATCH_SIZE):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = Config.CLASS_NAMES
        
    def scan_directory(self, split_name):
        """Recursively collect file paths and class labels from a split folder."""
        split_dir = os.path.join(self.data_dir, split_name)
        filepaths = []
        labels = []
        for class_idx, class_name in enumerate(self.class_names):
            class_folder = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            for fname in os.listdir(class_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepaths.append(os.path.join(class_folder, fname))
                    labels.append(class_idx)
        return np.array(filepaths), np.array(labels)

    def load_all_data(self):
        """Load file paths for train, val, and test sets and re-combine train+val for proper stratification."""
        train_fps, train_lbls = self.scan_directory("train")
        val_fps, val_lbls = self.scan_directory("val")
        test_fps, test_lbls = self.scan_directory("test")
        
        # Combine train and val (val in raw dataset only had 16 images total)
        all_train_fps = np.concatenate([train_fps, val_fps])
        all_train_lbls = np.concatenate([train_lbls, val_lbls])
        
        return all_train_fps, all_train_lbls, test_fps, test_lbls

    def compute_class_weights(self, labels):
        """Calculate balanced class weights to compensate for class imbalance."""
        classes = np.unique(labels)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
        return dict(zip(classes, weights))

    def get_augmentation_layer(self):
        """Return data augmentation layer."""
        return keras.Sequential([
            keras.layers.RandomFlip("horizontal", seed=Config.SEED),
            keras.layers.RandomRotation(0.1, seed=Config.SEED),
            keras.layers.RandomZoom(0.1, seed=Config.SEED),
        ], name="data_augmentation")

    def _parse_image(self, filepath, label, preprocess_fn=None):
        """Parse image file into a tensor with preprocessing."""
        img_bytes = tf.io.read_file(filepath)
        img = tf.image.decode_jpeg(img_bytes, channels=Config.CHANNELS)
        img = tf.image.resize(img, self.img_size)
        if preprocess_fn is not None:
            img = preprocess_fn(img)
        else:
            img = img / 255.0  # Default scaling to [0, 1]
        return img, label

    def create_tf_dataset(self, filepaths, labels, shuffle=True, augment=False, preprocess_fn=None):
        """Build tf.data.Dataset with caching, batching, augmentation, and prefetching."""
        ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
        
        if shuffle:
            ds = ds.shuffle(buffer_size=len(filepaths), seed=Config.SEED)
            
        def parse_fn(fp, lbl):
            return self._parse_image(fp, lbl, preprocess_fn=preprocess_fn)
            
        ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        if augment:
            aug_layer = self.get_augmentation_layer()
            ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
            
        ds = ds.batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    def get_stratified_kfold_splits(self, n_splits=5):
        """Generate Stratified K-Fold train and validation splits."""
        train_fps, train_lbls, _, _ = self.load_all_data()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.SEED)
        
        splits = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_fps, train_lbls)):
            splits.append({
                "fold": fold + 1,
                "train_fps": train_fps[train_idx],
                "train_lbls": train_lbls[train_idx],
                "val_fps": train_fps[val_idx],
                "val_lbls": train_lbls[val_idx],
            })
        return splits
