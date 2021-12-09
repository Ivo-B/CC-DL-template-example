from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


def read_data_fn(image: str, label: int) -> (tf.Tensor, tf.Tensor):
    """Reading data with numpy.

    :param image:
    :param label:
    :return:
    """
    image = np.load(image)[..., np.newaxis]
    return image.astype(np.float32), tf.cast(label, tf.uint8)


def load_data_fn(image: str, label: int) -> (tf.Tensor, tf.Tensor):
    """TF helper function for loading data with numpy.

    :param image:
    :param label:
    :return:
    """
    return tf.numpy_function(
        func=read_data_fn,
        inp=[image, label],
        Tout=[tf.float32, tf.uint8],
    )


def read_data_pair_fn(image: str, mask: str) -> (tf.Tensor, tf.Tensor):
    """Reading data with numpy.

    :param image:
    :param mask:
    :return:
    """
    image = np.load(image)
    mask = np.load(mask)
    return tf.cast(image, tf.float32), tf.cast(mask, tf.uint8)


def load_data_pair_fn(image: str, mask: str) -> (tf.Tensor, tf.Tensor):
    """TF helper function for loading data with numpy.

    :param image:
    :param mask:
    :return:
    """
    return tf.numpy_function(
        func=read_data_pair_fn,
        inp=[image, mask],
        Tout=[tf.float32, tf.uint8],
    )


class TfDataloader(ABC):
    """Abstract class for tf dataset definitions."""

    aug_comp_train = None
    aug_comp_val = None

    # These operations have to be implemented in subclasses.
    @abstractmethod
    def load_data(self, data_list: str, do_shuffle: bool) -> tuple[np.array, np.array]:
        """Implement data loading.

        :param data_list:
        :param do_shuffle:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def get_tf_dataset(self, phase: str) -> tf.data.Dataset:
        """Used to return a full tf.dataset pipeline.

        :param phase:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def set_aug_train(self, aug_comp: list):
        """Setter for class variable `aug_comp_train`."""
        raise NotImplementedError()

    @abstractmethod
    def set_aug_val(self, aug_comp: list):
        """Setter for class variable `aug_comp_val`."""
        raise NotImplementedError()
