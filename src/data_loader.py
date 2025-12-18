import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# Configuration

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ImageDataGenerators

def get_train_datagen():
    return ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True
    )

def get_val_test_datagen():
    return ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

# Data Loaders

def load_train_data():
    train_datagen = get_train_datagen()
    return train_datagen.flow_from_directory(
        DATA_DIR / "train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        seed=SEED
    )

def load_val_data():
    val_datagen = get_val_test_datagen()
    return val_datagen.flow_from_directory(
        DATA_DIR / "val",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

def load_test_data():
    test_datagen = get_val_test_datagen()
    return test_datagen.flow_from_directory(
        DATA_DIR / "test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

def get_data_loaders():
    train_gen = load_train_data()
    val_gen = load_val_data()
    test_gen = load_test_data()
    return train_gen, val_gen, test_gen
