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


class TfDataloader(ABC):
    """
    The Abstract Class defines a template method that contains a skeleton of
    some algorithm, composed of calls to (usually) abstract primitive
    operations.

    Concrete subclasses should implement these operations, but leave the
    template method itself intact.
    """
    aug_comp_train = None
    aug_comp_val = None

    # These operations have to be implemented in subclasses.
    @abstractmethod
    def load_data(self, data_list: str, do_shuffle: bool) -> tuple[np.array, np.array]:
        pass  # noqa: WPS420

    @abstractmethod
    def get_tf_dataset(self, phase: str) -> tf.data.Dataset:
        pass  # noqa: WPS420

    @abstractmethod
    def set_aug_train(self):
        pass  # noqa: WPS420

    @abstractmethod
    def set_aug_val(self):
        pass  # noqa: WPS420
