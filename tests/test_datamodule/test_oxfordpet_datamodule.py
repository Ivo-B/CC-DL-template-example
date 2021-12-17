import os
from pathlib import Path

import pytest
import dotenv

import tensorflow as tf
from cctest.datamodule.oxfordpet_datamodule import OxfordPetDataset
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

DATA_DIR = Path(os.environ['PROJECT_DIR']) / "data"
TRAIN_LIST = "OxfordPet_training_data.txt"
VAL_LIST = "OxfordPet_validation_data.txt"
TEST_LIST = "OxfordPet_testing_data.txt"


@pytest.mark.parametrize(
    "batch_size, num_gpus, expected", [
        (32, 0, 32),
        (64, 1, 64),
        (16, 4, 64),
    ])
def test_oxfordpet_datamodule(data_aug_oxfordpet, batch_size, num_gpus, expected):
    datamodule = OxfordPetDataset(
        data_dir=str(DATA_DIR),
        train_list=TRAIN_LIST,
        val_list=VAL_LIST,
        test_list=TEST_LIST,
        data_aug=data_aug_oxfordpet,
        batch_size=batch_size,
        num_gpus=num_gpus,
        cache_data=False,
    )

    assert datamodule._global_batch_size == expected
    assert datamodule._train_data_size == 4_631
    assert datamodule._val_data_size == 514
    assert datamodule._test_data_size == 2_204

    assert datamodule._train_data_size + datamodule._val_data_size + datamodule._test_data_size == 7_349

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

    assert len(x) == expected
    assert x.shape[1:] == datamodule._img_shape
    assert x.dtype == tf.float32
    assert tf.math.reduce_min(x) >= -0.5
    assert tf.math.reduce_max(x) <= 0.5

    assert len(y) == expected
    assert y.shape[1:-1] == datamodule._img_shape[:-1]
    assert y.shape[-1] == datamodule._n_classes
    assert y.dtype == tf.float32
