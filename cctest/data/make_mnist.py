import os
from pathlib import Path

import numpy as np
import ray
from dotenv import find_dotenv, load_dotenv
from keras.utils.data_utils import get_file
from natsort import natsorted
from sklearn.model_selection import StratifiedShuffleSplit

# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())


def load_annotation_data(root_path):
    with np.load(root_path, allow_pickle=True) as f:  # pylint: disable=unexpected-keyword-arg
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    all_x = np.concatenate((x_train, x_test))
    all_y = np.concatenate((y_train, y_test))
    return all_x, all_y


@ray.remote
def ray_preprocessing(full_dataset, block_idxs, output_path):
    # Do some image processing.
    out_results = []
    for idx in block_idxs:
        img = full_dataset[idx, ...] / 255.0
        np.save(output_path / f"{idx:05d}", img)
        out_results.append(f"{idx:05d}.npy")
    return out_results


def pre_process_data(full_dataset, num_cpus):
    output_path = Path(os.environ.get("PROJECT_DIR")) / "data" / "processed" / "mnist"
    os.makedirs(output_path, exist_ok=True)

    full_dataset_id = ray.put(full_dataset)
    block_idxs = np.array_split(np.arange(len(full_dataset)), num_cpus)
    result_ids = [ray_preprocessing.remote(full_dataset_id, block_idxs[x], output_path) for x in range(num_cpus)]

    file_names = []
    while len(result_ids):
        done_id, result_ids = ray.wait(result_ids)
        file_names += ray.get(done_id[0])
    file_names = natsorted(file_names)
    return np.array(file_names)


def train_val_test_spitting(file_names, all_y):
    output_path = Path(os.environ.get("PROJECT_DIR")) / "data"
    # First split of all data into training+validation and testing
    sss = StratifiedShuffleSplit(n_splits=1, test_size=10000, random_state=42)
    for train_index, test_index in sss.split(file_names, all_y):
        X_train_val, X_test = file_names[train_index], file_names[test_index]
        y_train_val, y_test = all_y[train_index], all_y[test_index]

    with open(output_path / "MNIST_testing_data.txt", "w") as f:
        for file_name, y in zip(X_test, y_test):
            f.write(f"{file_name}, {y}\n")

    # Second split of "training+validation" into training and validation
    sss = StratifiedShuffleSplit(n_splits=1, test_size=5000, random_state=42)
    for train_index, test_index in sss.split(X_train_val, y_train_val):
        X_train, X_val = X_train_val[train_index], X_train_val[test_index]
        y_train, y_val = y_train_val[train_index], y_train_val[test_index]

    with open(output_path / "MNIST_training_data.txt", "w") as f:
        for file_name, y in zip(X_train, y_train):
            f.write(f"{file_name}, {y}\n")

    with open(output_path / "MNIST_validation_data.txt", "w") as f:
        for file_name, y in zip(X_val, y_val):
            f.write(f"{file_name}, {y}\n")


if __name__ == "__main__":
    root_path = Path(os.environ.get("PROJECT_DIR")) / "data" / "raw" / "mnist"
    ray.init()

    print(f"Creating folder for download: {root_path}")
    os.makedirs(root_path, exist_ok=True)
    file_name = "mnist.npz"
    origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    path_download = get_file(
        fname=file_name,
        cache_dir=Path(os.environ.get("PROJECT_DIR")) / "data" / "raw",
        cache_subdir="mnist",
        origin=origin_folder + file_name,
        file_hash="731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1",
    )

    print(f"Load data and pre-process it.")
    all_x, all_y = load_annotation_data(path_download)
    file_names = pre_process_data(all_x, os.cpu_count())

    print(f"Creating training, validation, and test split.")
    train_val_test_spitting(file_names, all_y)
    ray.shutdown()
