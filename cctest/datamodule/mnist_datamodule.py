from typing import Optional, Tuple

from pathlib import Path
from functools import partial
import numpy as np
import tensorflow as tf
import albumentations as A

from .base_datamodule import TfDataloader
from ..utils import utils

log = utils.get_logger(__name__)

AUTOTUNE = tf.data.experimental.AUTOTUNE


class MNISTDataset(TfDataloader):
    def __init__(self,
                 data_dir: str,
                 data_training_list: str,
                 data_val_list: str,
                 data_test_list: str,
                 batch_size: int,
                 num_gpus: int
                 ):
        self.data_dir = Path(data_dir)
        self.data_training_list = data_training_list
        self.data_val_list = data_val_list
        if num_gpus == 0:
            num_gpus = 1
        self.global_batch_size = batch_size * num_gpus
        self.n_classes = 10
        self.img_shape = (28, 28, 1)

        # loading file path from text file
        self.img_train_paths, self.train_labels = self.load_data(data_training_list)
        self.img_val_paths, self.val_labels = self.load_data(data_val_list)
        self.set_aug_train()
        self.set_aug_val()
        self.val_data_size = len(self.img_val_paths)
        self.train_data_size = len(self.img_train_paths)
        self.steps_per_epoch = self.train_data_size // self.global_batch_size

    def load_data(self, data_list: str, do_shuffle: bool = True) -> tuple[np.array, np.array]:
        x_data: list = []
        y_data: list = []
        with open(self.data_dir / data_list, 'r') as file:
            for line in file.readlines():
                file_path: str
                label: str
                [file_path, label] = line.split(',')
                x_data.append(str(self.data_dir / 'processed' / file_path))
                y_data.append(int(label))

        log.debug(f'x_len, y_len: {len(x_data)}, {len(y_data)}')
        if do_shuffle:
            # shuffle training
            indexes = np.arange(len(x_data))
            np.random.shuffle(indexes)
            x_data = np.array(x_data)[indexes]
            y_data = np.array(y_data)[indexes]
        return x_data, y_data

    def get_tf_dataset(self, phase: str) -> tf.data.Dataset:
        if phase == 'training':
            img_paths, labels = self.img_train_paths, self.train_labels
        elif phase == 'validation':
            img_paths, labels = self.img_val_paths, self.val_labels
        else:
            raise ValueError
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))

        if phase == 'training':
            dataset = dataset.shuffle(len(img_paths), reshuffle_each_iteration=True, seed=42)

        dataset = dataset.map(MNISTDataset.load_data_fn, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)

        dataset = dataset.map(partial(MNISTDataset.process_data_fn,
                                      phase=phase,
                                      img_shape=self.img_shape,
                                      num_classes=self.n_classes),
                              num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)

        if phase == 'training':
            dataset = dataset.repeat()
        dataset = dataset.batch(self.global_batch_size, drop_remainder=True).prefetch(buffer_size=AUTOTUNE)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
        return dataset

    @staticmethod
    def set_aug_train():
        aug_comp: list = [A.Flip(p=0.2),
                          A.Rotate(limit=90, p=0.2),
                          A.RandomBrightnessContrast(p=0.2),
                          A.Normalize(mean=0.1307, std=0.3081, max_pixel_value=1.0)]

        MNISTDataset.__class__.aug_comp_train = A.Compose(aug_comp)

    @staticmethod
    def set_aug_val():
        aug_comp: list = [A.Normalize(mean=0.1307, std=0.3081, max_pixel_value=1.0)]

        MNISTDataset.__class__.aug_comp_val = A.Compose(aug_comp)

    @staticmethod
    def load_data_fn(image: str, label: int) -> (tf.Tensor, tf.Tensor):
        image, label = tf.numpy_function(func=MNISTDataset.read_data_fn, inp=[image, label],
                                         Tout=[tf.float32, tf.uint8])
        return image, label

    @staticmethod
    def read_data_fn(image: str, label: int) -> (tf.Tensor, tf.Tensor):
        image = np.load(image)[..., np.newaxis].astype(np.float32)
        return image, tf.cast(label, tf.uint8)

    @staticmethod
    def process_data_fn(image: tf.Tensor,
                        label: tf.Tensor,
                        phase: str,
                        img_shape: list,
                        num_classes: int) -> (tf.Tensor, tf.Tensor):
        if phase == 'training':
            aug_img = tf.numpy_function(func=MNISTDataset.aug_train_fn, inp=[image], Tout=tf.float32)
        elif phase == 'validation':
            aug_img = tf.numpy_function(func=MNISTDataset.aug_val_fn, inp=[image], Tout=tf.float32)
        else:
            raise ValueError

        aug_img.set_shape(img_shape)
        label.set_shape([])
        aug_label_one_hot = tf.one_hot(label, depth=num_classes)
        return aug_img, aug_label_one_hot

    @staticmethod
    def aug_train_fn(image: np.array) -> tf.Tensor:
        data = {"image": image}
        aug_data = MNISTDataset.__class__.aug_comp_train(**data)
        return tf.cast(aug_data["image"], tf.float32)

    @staticmethod
    def aug_val_fn(image: np.array) -> tf.Tensor:
        data = {"image": image}
        aug_data = MNISTDataset.__class__.aug_comp_train(**data)
        return tf.cast(aug_data["image"], tf.float32)
