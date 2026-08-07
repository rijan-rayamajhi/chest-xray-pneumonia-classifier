import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import Config

PREPROCESS_FUNCTIONS = {
    "BaselineCNN": None,
    "DenseNet121": keras.applications.densenet.preprocess_input,
    "ResNet50V2": keras.applications.resnet_v2.preprocess_input,
    "EfficientNetB0": keras.applications.efficientnet.preprocess_input,
    "MobileNetV2": keras.applications.mobilenet_v2.preprocess_input,
}

def get_preprocess_fn(model_name):
    """Return specific preprocessing function for each model architecture."""
    return PREPROCESS_FUNCTIONS.get(model_name, None)

def build_baseline_cnn(input_shape=(224, 224, 3)):
    """Build custom shallow CNN baseline architecture."""
    inputs = keras.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="final_conv_layer")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    
    model = keras.Model(inputs, outputs, name="BaselineCNN")
    return model

def build_transfer_model(architecture_name, input_shape=(224, 224, 3), dropout_rate=0.4, learning_rate=Config.LEARNING_RATE):
    """Factory function for pretrained backbone models (DenseNet121, ResNet50V2, EfficientNetB0, MobileNetV2)."""
    inputs = keras.Input(shape=input_shape)
    
    # Select backbone
    if architecture_name == "DenseNet121":
        base_model = keras.applications.DenseNet121(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
    elif architecture_name == "ResNet50V2":
        base_model = keras.applications.ResNet50V2(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
    elif architecture_name == "EfficientNetB0":
        base_model = keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
    elif architecture_name == "MobileNetV2":
        base_model = keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture_name}")
        
    # Freeze pretrained feature extraction layers initially
    base_model.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=architecture_name)
    return model, base_model

def get_target_conv_layer(model_name, model):
    """Return the name of the final convolutional layer for Grad-CAM explainability."""
    if model_name == "BaselineCNN":
        return "final_conv_layer"
    
    # For transfer learning models, locate last 4D convolutional layer
    for layer in reversed(model.layers):
        if isinstance(layer, keras.Model):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, (layers.Conv2D, layers.DepthwiseConv2D)):
                    return sub_layer.name
        if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D)):
            return layer.name
            
    # Default fallback names for standard keras applications
    defaults = {
        "DenseNet121": "relu",
        "ResNet50V2": "post_relu",
        "EfficientNetB0": "top_activation",
        "MobileNetV2": "out_relu",
    }
    return defaults.get(model_name, None)
