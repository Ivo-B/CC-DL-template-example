import os

import pytest
import dotenv
from omegaconf import DictConfig

import tensorflow as tf
from cctest.datamodule.oxford_iiit_pet_datamodule import OxfordPetDataset
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

DATA_DIR = f"{os.environ['PROJECT_DIR']}/data"
TRAIN_LIST = f"OxfordPet_training_data.txt"
VAL_LIST = f"OxfordPet_validation_data.txt"
TEST_LIST = f"OxfordPet_testing_data.txt"
DATA_AUG = DictConfig({
    "training": {
        "flip_horizontal": {
            "_target_": "albumentations.HorizontalFlip",
            "p": 0.2},
        "shift_scale_rotate": {
            "_target_": "albumentations.ShiftScaleRotate",
            "p": 0.2}
    },
    # will be added to both transformation pipelines (Training and Validation)
    "norm_data": {
        "_target_": "albumentations.Normalize",
        "mean": [0.5, 0.5, 0.5],
        "std": [1, 1, 1],
        "max_pixel_value": 1.0
    },
})


@pytest.mark.parametrize("batch_size, num_gpus, expected", [(32, 0, 32), (64, 1, 64), (16, 3, 48)])
def test_oxford_iiit_pet_datamodule(batch_size, num_gpus, expected):
    datamodule = OxfordPetDataset(
        data_dir=DATA_DIR,
        train_list=TRAIN_LIST,
        val_list=VAL_LIST,
        test_list=TEST_LIST,
        data_aug=DATA_AUG,
        batch_size=batch_size,
        num_gpus=num_gpus, )

    assert datamodule._global_batch_size == expected
    assert datamodule._img_train_paths and datamodule._mask_train_paths
    assert datamodule._img_val_paths and datamodule._mask_train_paths
    assert (len(datamodule._img_train_paths) + len(datamodule._img_val_paths) == 5659)

    training_dataset = datamodule.get_tf_dataset("training")
    validation_dataset = datamodule.get_tf_dataset("validation")
    assert training_dataset
    assert validation_dataset
    with pytest.raises(Exception):
        datamodule.get_tf_dataset("val")
    with pytest.raises(Exception):
        datamodule.get_tf_dataset("train")

    batch = next(iter(training_dataset))
    x, y = batch

    assert len(x) == batch_size
    assert x.shape[1:] == datamodule._img_shape
    assert x.dtype == tf.float32
    assert x.min() < 0

    assert len(y) == batch_size
    assert y.shape[1:-1] == datamodule._img_shape[:-1]
    assert y.shape[-1] == datamodule._n_classes
    assert y.dtype == tf.uint8
