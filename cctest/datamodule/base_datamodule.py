import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod


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
        pass

    @abstractmethod
    def get_tf_dataset(self, phase: str) -> tf.data.Dataset:
        pass

    @staticmethod
    @abstractmethod
    def load_data_fn(image: str, label: int) -> (tf.Tensor, tf.Tensor):
        pass

    @staticmethod
    @abstractmethod
    def read_data_fn(image: str, mask: str) -> (tf.Tensor, tf.Tensor):
        return image, mask

    @staticmethod
    @abstractmethod
    def process_data_fn(image: tf.Tensor,
                        label: tf.Tensor,
                        aug_comp: callable,
                        img_shape: list,
                        num_classes: int) -> (tf.Tensor, tf.Tensor):
        pass

    @staticmethod
    @abstractmethod
    def aug_train_fn(image: np.array) -> tf.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def aug_val_fn(image: np.array) -> tf.Tensor:
        pass



