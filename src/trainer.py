import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path

# Configuration

PHASE1_EPOCHS = 10
PHASE1_LR = 1e-3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

PHASE1_MODEL_PATH = str(MODEL_DIR / "mobilenetv2_phase1.h5")

# Phase-1 Trainer

def train_phase1(model, train_gen, val_gen):
    """
    Phase-1 training:
    - Freeze entire backbone
    - Train only classification head
    """

    # Ensure base model is frozen
    for layer in model.layers:
        layer.trainable = False

    # Re-enable trainability for classification head
    for layer in model.layers[-4:]:
        layer.trainable = True

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            PHASE1_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE1_EPOCHS,
        callbacks=callbacks
    )

    print("Phase-1 MobileNetV2 model saved")

    return model, history
