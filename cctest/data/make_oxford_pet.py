import os
from pathlib import Path

import numpy as np
from dotenv import find_dotenv, load_dotenv
from keras.utils.data_utils import get_file
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import StratifiedShuffleSplit

from cctest.utils.utils import get_logger

log = get_logger(__name__)

# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())


def download_and_extract_data(root_path, file_name: str):
    origin_folder = "http://www.robots.ox.ac.uk/~vgg/data/pets/data/"
    get_file(
        root_path / file_name,
        cache_dir=Path(os.environ.get("PROJECT_DIR")) / "data" / "raw",
        cache_subdir="oxford-pet",
        extract=True,
        origin=origin_folder + file_name,
    )
    os.remove(root_path / file_name)


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
    all_x = []
    all_y = []
    for idx in range(len(full_dataset)):
        all_x.append(full_dataset[idx]["file_name"])
        all_y.append(full_dataset[idx]["label"])
    all_x = np.array(all_x)
    all_y = np.array(all_y)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=2204, random_state=42)
    for train_index, test_index in sss.split(all_x, all_y):
        X_train_val, X_test = all_x[train_index], all_x[test_index]
        y_train_val, y_test = all_y[train_index], all_y[test_index]

    path = Path(os.environ.get("PROJECT_DIR")) / "data"
    with open(path / "OxfordPet_test_data.txt", "a") as f:
        for name, y in zip(X_test, y_test):
            f.write(f"{name}.npy, {y}\n")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=514, random_state=42)
    for train_index, test_index in sss.split(X_train_val, y_train_val):
        X_train, X_val = all_x[train_index], all_x[test_index]
        y_train, y_val = all_y[train_index], all_y[test_index]

    with open(path / "OxfordPet_training_data.txt", "a") as f:
        for name, y in zip(X_train, y_train):
            f.write(f"{name}.npy, {y}\n")

    with open(path / "OxfordPet_validation_data.txt", "a") as f:
        for name, y in zip(X_val, y_val):
            f.write(f"{name}.npy, {y}\n")


if __name__ == "__main__":
    root_path = Path(os.environ.get("PROJECT_DIR")) / "data" / "raw" / "oxford-pet"
    os.makedirs(root_path, exist_ok=True)

    download_and_extract_data(root_path, "images.tar.gz")
    download_and_extract_data(root_path, "annotations.tar.gz")

    full_dataset = load_annotation_data(root_path)
    pre_process_data(full_dataset)
    train_val_test_spitting(full_dataset)
