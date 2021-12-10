from functools import partial
from pathlib import Path

import hydra
import numpy as np
import tensorflow as tf
from albumentations import BasicTransform, Compose
from omegaconf import DictConfig

from {{cookiecutter.module_name}}.datamodule.base_datamodule import TfDataloader, load_data_fn
from {{cookiecutter.module_name}}.utils import utils

log = utils.get_logger(__name__)

AUTOTUNE = tf.data.experimental.AUTOTUNE


def aug_train_fn(image: np.array) -> tf.Tensor:
    """Helper function to apply training augmentation.

    :param image:
    :return:
    """
    input_data = {"image": image}
    aug_data = MNISTDataset.aug_comp_train(**input_data)
    return tf.cast(aug_data["image"], tf.float32)


def aug_val_fn(image: np.array) -> tf.Tensor:
    """Helper function to apply training augmentation.

    :param image:
    :return:
    """
    input_data = {"image": image}
    aug_data = MNISTDataset.aug_comp_val(**input_data)
    return tf.cast(aug_data["image"], tf.float32)


def process_data_fn(
    image: tf.Tensor,
    label: tf.Tensor,
    phase: str,
    img_shape: list,
    num_classes: int,
) -> (tf.Tensor, tf.Tensor):
    """Pipeline for preprocessing data.

    :param image:
    :param label:
    :param phase:
    :param img_shape:
    :param num_classes:
    :return:
    """
    if phase == "training":
        aug_img = tf.numpy_function(
            func=aug_train_fn,
            inp=[image],
            Tout=tf.float32,
        )
    elif phase == "validation":
        aug_img = tf.numpy_function(
            func=aug_val_fn,
            inp=[image],
            Tout=tf.float32,
        )
    else:
        raise ValueError

    aug_img.set_shape(img_shape)
    label.set_shape([])
    aug_label_one_hot = tf.one_hot(label, depth=num_classes)
    return aug_img, aug_label_one_hot


class MNISTDataset(TfDataloader):
    """MNIST dataset call."""

    def __init__(
        self,
        data_dir: str,
        data_training_list: str,
        data_val_list: str,
        data_test_list: str,
        batch_size: int,
        num_gpus: int,
        data_aug: DictConfig,
    ):
        """__init__.

        :param data_dir:
        :param data_training_list:
        :param data_val_list:
        :param data_test_list:
        :param batch_size:
        :param num_gpus:
        :param data_aug:
        """
        self._data_dir = Path(data_dir)
        self._data_training_list = data_training_list
        self._data_val_list = data_val_list
        if num_gpus == 0:
            num_gpus = 1
        self._global_batch_size = batch_size * num_gpus
        self._n_classes = 10
        self._img_shape = (28, 28, 1)

        # loading file path from text file
        img_train_paths, train_labels = self.load_data(
            data_training_list,
        )
        self._img_train_paths = img_train_paths
        self._train_labels = train_labels
        img_val_paths, val_labels = self.load_data(data_val_list)
        self._img_val_paths = img_val_paths
        self._val_labels = val_labels

        aug_comp_training: list[BasicTransform] = []
        aug_comp_validation: list[BasicTransform] = []
        if data_aug:
            if data_aug.get("training"):
                for _, da_conf in data_aug.training.items():
                    if "_target_" in da_conf:
                        log.info(f"Instantiating training data transformation <{da_conf._target_}>")
                        aug_comp_training.append(hydra.utils.instantiate(da_conf))

            if data_aug.get("validation"):
                for _, da_conf in data_aug.validation.items():
                    if "_target_" in da_conf:
                        log.info(f"Instantiating validation data transformation <{da_conf._target_}>")
                        aug_comp_validation.append(hydra.utils.instantiate(da_conf))

            for da_key, da_conf in data_aug.items():
                if "_target_" in da_conf:
                    log.info(f"Instantiating Data Transformation <{da_conf._target_}>")
                    transformation = hydra.utils.instantiate(da_conf)
                    aug_comp_training.append(transformation)
                    aug_comp_validation.append(transformation)

        self.set_aug_train(aug_comp_training)
        self.set_aug_val(aug_comp_validation)
        self._val_data_size = len(self._img_val_paths)
        self._train_data_size = len(self._img_train_paths)
        self.steps_per_epoch = self._train_data_size // self._global_batch_size

    def load_data(
        self,
        data_list: str,
        do_shuffle: bool = True,
    ) -> tuple[np.array, np.array]:
        """load_data.

        :param data_list:
        :param do_shuffle:
        :return:
        """
        x_data: list = []
        y_data: list = []
        with open(self._data_dir / data_list, "r") as file_stream:
            for line in file_stream.readlines():
                file_path: str
                label: str
                file_path, label = line.split(",")
                x_data.append(str(self._data_dir / "processed/mnist" / file_path))
                y_data.append(int(label))
        log.debug("x_len, y_len: %d, %d", len(x_data), len(y_data))
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        if do_shuffle:
            # shuffle training
            indexes = np.arange(len(x_data))
            np.random.shuffle(indexes)
            x_data = x_data[indexes]
            y_data = y_data[indexes]
        return x_data, y_data

    def set_aug_train(self, aug_comp):
        """Sets training augmentation.

        :return:
        """
        self.__class__.aug_comp_train = Compose(aug_comp)

    def set_aug_val(self, aug_comp):
        """Sets validation augmentation.

        :return:
        """
        self.__class__.aug_comp_val = Compose(aug_comp)

    def get_tf_dataset(self, phase: str) -> tf.data.Dataset:
        """Creates and returns full tf dataset.

        :param phase:
        :return:
        """
        if phase == "training":
            img_paths, labels = self._img_train_paths, self._train_labels
        elif phase == "validation":
            img_paths, labels = self._img_val_paths, self._val_labels
        else:
            raise ValueError
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))

        if phase == "training":
            dataset = dataset.shuffle(
                len(img_paths),
                reshuffle_each_iteration=True,
                seed=42,
            )

        dataset = dataset.map(
            load_data_fn,
            num_parallel_calls=AUTOTUNE,
        ).prefetch(buffer_size=AUTOTUNE)

        dataset = dataset.map(
            partial(
                process_data_fn,
                phase=phase,
                img_shape=self._img_shape,
                num_classes=self._n_classes,
            ),
            num_parallel_calls=AUTOTUNE,
        ).prefetch(buffer_size=AUTOTUNE)

        if phase == "training":
            dataset = dataset.repeat()
        dataset = dataset.batch(
            self._global_batch_size,
            drop_remainder=True,
        ).prefetch(buffer_size=AUTOTUNE)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        return dataset.with_options(options)
