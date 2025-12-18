import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

# Configuration

IMG_SIZE = (224, 224, 3)
LEARNING_RATE = 1e-3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Model Builder

def build_mobilenetv2_transfer_model():
    """
    Build MobileNetV2-based transfer learning model
    (Feature Extraction phase: base frozen)
    """

    # Load pre-trained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE,
        include_top=False,
        weights="imagenet"
    )

    # Freeze convolutional base
    base_model.trainable = False

    # Custom classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model
