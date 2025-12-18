import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

# Configuration

IMG_SIZE = (224, 224, 3)
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Model Definition

def build_baseline_cnn():
    model = models.Sequential([
        layers.Input(shape=IMG_SIZE),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Training Function

def train_baseline(train_gen, val_gen):
    model = build_baseline_cnn()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    model.save(MODEL_DIR / "baseline_cnn.h5")
    print("Baseline CNN model saved to models/baseline_cnn.h5")

    return model, history
