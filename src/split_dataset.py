import os
import shutil
import random
from pathlib import Path

BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "train"

SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

CLASSES = ["cats", "dogs"]
RANDOM_SEED = 42


def create_dirs():
    """
    Create train/val/test directories with class subfolders.
    """
    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(BASE_DIR / split / cls, exist_ok=True)


def split_images(images, split_ratios):
    """
    Split a list of images according to given ratios.
    """
    total = len(images)
    train_end = int(total * split_ratios["train"])
    val_end = train_end + int(total * split_ratios["val"])

    return {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }


def copy_images(split_map, class_name):
    """
    Copy images into corresponding split directories.
    """
    for split, files in split_map.items():
        for filename in files:
            src = RAW_DIR / filename
            dst = BASE_DIR / split / class_name / filename

            # Avoid duplicate copies if script is re-run
            if not dst.exists():
                shutil.copy(src, dst)

def split_dataset():
    random.seed(RANDOM_SEED)

    # Collect only valid image files
    cat_images = [
        f for f in os.listdir(RAW_DIR)
        if f.startswith("cat") and (RAW_DIR / f).is_file()
    ]

    dog_images = [
        f for f in os.listdir(RAW_DIR)
        if f.startswith("dog") and (RAW_DIR / f).is_file()
    ]

    random.shuffle(cat_images)
    random.shuffle(dog_images)

    cat_splits = split_images(cat_images, SPLITS)
    dog_splits = split_images(dog_images, SPLITS)

    copy_images(cat_splits, "cats")
    copy_images(dog_splits, "dogs")


def print_summary():
    """
    Print final image counts for verification.
    """
    print("\nDataset split summary:")
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            path = BASE_DIR / split / cls
            print(f"{split}/{cls}: {len(os.listdir(path))} images")

if __name__ == "__main__":
    create_dirs()
    split_dataset()
    print_summary()
    print("\nDataset successfully split into train / val / test")
