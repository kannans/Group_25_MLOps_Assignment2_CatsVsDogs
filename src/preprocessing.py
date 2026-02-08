import os
import shutil
import random
from torchvision import transforms, datasets
from PIL import Image
import torch


def create_dummy_data(data_dir, num_images=20):
    """Creates dummy cats and dogs images for testing the pipeline."""
    os.makedirs(os.path.join(data_dir, "cats"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "dogs"), exist_ok=True)

    for i in range(num_images):
        # Create a random image
        img = Image.new(
            "RGB",
            (224, 224),
            color=(
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            ),
        )
        label = "cats" if i % 2 == 0 else "dogs"
        img.save(os.path.join(data_dir, label, f"image_{i}.jpg"))

    print(f"Created {num_images} dummy images in {data_dir}")


def preprocess_data(
    raw_dir, processed_dir, img_size=(224, 224), split_ratios=(0.8, 0.1, 0.1)
):
    """
    Physically splits the data into train, val, and test directories.
    """
    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        print(f"Raw data directory {raw_dir} is empty. Creating dummy data...")
        create_dummy_data(raw_dir)

    classes = [
        d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))
    ]

    for cls in classes:
        cls_raw_dir = os.path.join(raw_dir, cls)
        images = [
            f
            for f in os.listdir(cls_raw_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])

        splits = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        for split_name, split_images in splits.items():
            split_dir = os.path.join(processed_dir, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img_name in split_images:
                shutil.copy2(
                    os.path.join(cls_raw_dir, img_name),
                    os.path.join(split_dir, img_name),
                )

    print(f"Data split completed. Processed data saved in {processed_dir}")


if __name__ == "__main__":
    RAW_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"

    # Ensure processed dir is clean
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR)

    preprocess_data(RAW_DIR, PROCESSED_DIR)
