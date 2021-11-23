import os
from pathlib import Path

import numpy as np
from dotenv import find_dotenv, load_dotenv
from keras.utils.data_utils import get_file
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import StratifiedShuffleSplit

# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())


def download_and_extract_data(root_path, file_name: str):
    origin_folder = "http://www.robots.ox.ac.uk/~vgg/data/pets/data/"
    get_file(
        fname=file_name,
        cache_dir=Path(os.environ.get("PROJECT_DIR")) / "data" / "raw",
        cache_subdir="oxford-pet",
        extract=True,
        origin=origin_folder + file_name,
    )
    # os.remove(root_path / file_name)


def load_annotation_data(root_path):
    all_data = []
    with open(root_path / "annotations" / "list.txt", "r") as images_list:
        for line in images_list.readlines():
            if "#" in line:
                continue
            image_name, label, species, _ = line.strip().split(" ")
            trimaps_dir_path = os.path.join(root_path, "annotations", "trimaps")

            label = int(label) - 1
            species = int(species) - 1

            trimap_name = image_name + ".png"
            record = {
                "image": os.path.join(root_path, "images", image_name + ".jpg"),
                "label": label,
                "species": species,
                "file_name": image_name,
                "segmentation_mask": os.path.join(trimaps_dir_path, trimap_name),
            }
            all_data.append(record)
    return all_data


def pre_process_data(full_dataset):
    output_path = Path(os.environ.get("PROJECT_DIR")) / "data" / "processed" / "oxford-pet"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path / "images", exist_ok=True)
    os.makedirs(output_path / "masks", exist_ok=True)

    for idx in range(len(full_dataset)):
        # load
        img = io.imread(full_dataset[idx]["image"])
        img = img / 255
        img = resize(img, (128, 128), order=1, anti_aliasing=True).astype(np.float32)
        # remove alpha
        if img.shape[-1] == 4:
            img = img[..., :3]
        np.save(output_path / "images" / f'{full_dataset[idx]["file_name"]}', img)

        mask = io.imread(full_dataset[idx]["segmentation_mask"])
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        mask = resize(mask, (128, 128), order=0, preserve_range=True, anti_aliasing=False)
        np.save(output_path / "masks" / f'{full_dataset[idx]["file_name"]}', mask)


def train_val_test_spitting(full_dataset):
    output_path = Path(os.environ.get("PROJECT_DIR")) / "data"
    all_x = []
    all_y = []
    for idx in range(len(full_dataset)):
        all_x.append(full_dataset[idx]["file_name"])
        all_y.append(full_dataset[idx]["label"])
    all_x = np.array(all_x)
    all_y = np.array(all_y)

    # First split of all data into training+validation and testing
    sss = StratifiedShuffleSplit(n_splits=1, test_size=2204, random_state=42)
    for train_index, test_index in sss.split(all_x, all_y):
        X_train_val, X_test = all_x[train_index], all_x[test_index]
        y_train_val, y_test = all_y[train_index], all_y[test_index]

    with open(output_path / "OxfordPet_test_data.txt", "w") as f:
        for name, y in zip(X_test, y_test):
            f.write(f"{name}.npy, {y}\n")

    # Second split of "training+validation" into training and validation
    sss = StratifiedShuffleSplit(n_splits=1, test_size=514, random_state=42)
    for train_index, test_index in sss.split(X_train_val, y_train_val):
        X_train, X_val = X_train_val[train_index], X_train_val[test_index]
        y_train, y_val = y_train_val[train_index], y_train_val[test_index]

    with open(output_path / "OxfordPet_training_data.txt", "w") as f:
        for name, y in zip(X_train, y_train):
            f.write(f"{name}.npy, {y}\n")

    with open(output_path / "OxfordPet_validation_data.txt", "w") as f:
        for name, y in zip(X_val, y_val):
            f.write(f"{name}.npy, {y}\n")


if __name__ == "__main__":
    root_path = Path(os.environ.get("PROJECT_DIR")) / "data" / "raw" / "oxford-pet"

    print(f"Creating folder for download: {root_path}")
    os.makedirs(root_path, exist_ok=True)
    download_and_extract_data(root_path, "images.tar.gz")
    download_and_extract_data(root_path, "annotations.tar.gz")

    print(f"Load data and pre-process it.")
    full_dataset = load_annotation_data(root_path)
    pre_process_data(full_dataset)

    print(f"Creating training, validation, and test split.")
    train_val_test_spitting(full_dataset)
