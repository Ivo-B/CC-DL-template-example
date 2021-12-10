from functools import partial
from pathlib import Path

import hydra
import numpy as np
import tensorflow as tf
from albumentations import BasicTransform, Compose
from omegaconf import DictConfig

from cctest.datamodule.base_datamodule import TfDataloader, load_data_pair_fn
from cctest.utils import utils

log = utils.get_logger(__name__)

AUTOTUNE = tf.data.experimental.AUTOTUNE
SHUFFEL_SEED = 42


_LABEL_CLASSES = [
    "Abyssinian",
    "american_bulldog",
    "american_pit_bull_terrier",
    "basset_hound",
    "beagle",
    "Bengal",
    "Birman",
    "Bombay",
    "boxer",
    "British_Shorthair",
    "chihuahua",
    "Egyptian_Mau",
    "english_cocker_spaniel",
    "english_setter",
    "german_shorthaired",
    "great_pyrenees",
    "havanese",
    "japanese_chin",
    "keeshond",
    "leonberger",
    "Maine_Coon",
    "miniature_pinscher",
    "newfoundland",
    "Persian",
    "pomeranian",
    "pug",
    "Ragdoll",
    "Russian_Blue",
    "saint_bernard",
    "samoyed",
    "scottish_terrier",
    "shiba_inu",
    "Siamese",
    "Sphynx",
    "staffordshire_bull_terrier",
    "wheaten_terrier",
    "yorkshire_terrier",
]
_SPECIES_CLASSES = ["Cat", "Dog"]

PHASE_TRAIN = "training"
PHASE_VAL = "validation"
PHASE_TEST = "testing"
EVALUATION_PHASE = ("training", "validation", "testing")
DataPair = (tf.Tensor, tf.Tensor)


def aug_train_fn(image: np.array, mask: np.array) -> DataPair:
    """Helper function to apply training augmentation.

    :param mask:
    :param image:
    :return:
    """
    input_data = {"image": image, "mask": mask}
    aug_img, aug_mask = OxfordPetDataset.aug_comp_train(**input_data).values()
    return tf.cast(aug_img, tf.float32), tf.cast(aug_mask, tf.uint8)


def aug_val_fn(image: np.array, mask: np.array) -> DataPair:
    """Helper function to apply training augmentation.

    :param mask:
    :param image:
    :return:
    """
    input_data = {"image": image, "mask": mask}
    aug_img, aug_mask = OxfordPetDataset.aug_comp_val(**input_data).values()
    return tf.cast(aug_img, tf.float32), tf.cast(aug_mask, tf.uint8)


def process_data_fn(image: tf.Tensor, mask: tf.Tensor, phase: str, img_shape: list, num_classes: int) -> DataPair:
    """Pipeline for preprocessing data.

    :param image:
    :param mask:
    :param phase:
    :param img_shape:
    :param num_classes:
    :return:
    """
    if phase == PHASE_TRAIN:
        aug_img, aug_mask = tf.numpy_function(
            func=aug_train_fn,
            inp=[image, mask],
            Tout=[tf.float32, tf.uint8],
        )
    elif phase == PHASE_VAL:
        aug_img, aug_mask = tf.numpy_function(
            func=aug_val_fn,
            inp=[image, mask],
            Tout=[tf.float32, tf.uint8],
        )
    else:
        raise ValueError

    aug_img.set_shape(img_shape)
    aug_mask_one_hot = tf.one_hot(aug_mask, depth=num_classes)
    aug_mask_one_hot.set_shape(img_shape[:2] + (num_classes,))
    return aug_img, aug_mask_one_hot


def shuffle_data(data_pair):
    """Random shuffle a data pair.

    :param data_pair:
    :return:
    """
    # shuffle training
    indexes = np.arange(data_pair.shape[1])
    np.random.shuffle(indexes)
    data_pair[0, ...] = data_pair[0, indexes]
    data_pair[1, ...] = data_pair[1, indexes]
    return data_pair


class OxfordPetDataset(TfDataloader):
    """Oxford-IIIT pet dataset call.

    The Oxford-IIIT pet dataset is a 37 category pet image dataset with roughly 200 images for each class.
    The images have large variations in scale, pose and lighting. All images have an associated ground
    truth annotation of breed.
    """

    def __init__(
        self,
        data_dir: str,
        train_list: str,
        val_list: str,
        test_list: str,
        batch_size: int,
        num_gpus: int,
        data_aug: DictConfig,
    ):  # noqa: WPS211, E501
        """__init__.

        :param data_dir:
        :param train_list:
        :param val_list:
        :param test_list:
        :param batch_size:
        :param num_gpus:
        """
        self._data_dir = Path(data_dir)
        self._data_training_list = train_list
        self._data_val_list = val_list
        self._data_test_list = test_list
        if num_gpus == 0:
            num_gpus = 1
        self._global_batch_size = batch_size * num_gpus
        self._n_classes = 2
        self._img_shape = (128, 128, 3)

        # loading file path from text file
        img_train_paths, mask_train_paths = self.load_data(self._data_training_list)
        self._img_train_paths = img_train_paths
        self._mask_train_paths = mask_train_paths
        img_val_paths, mask_val_paths = self.load_data(self._data_val_list)
        self._img_val_paths = img_val_paths
        self._mask_val_paths = mask_val_paths

        aug_comp_training: list[BasicTransform] = []
        aug_comp_validation: list[BasicTransform] = []
        if data_aug:
            if data_aug.get("training"):
                for _, da_conf in data_aug.training.items():
                    if "_target_" in da_conf:
                        log.info(f"Instantiating training data transformation <{da_conf._target_}>")
                        aug_comp_training.append(hydra.utils.instantiate(da_conf))

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

    def load_data(self, data_list: str, do_shuffle: bool = True) -> tuple[np.array, np.array]:
        """load_data.

        :param data_list:
        :param do_shuffle:
        :return:
        """
        data_pair = [[], []]

        with open(self._data_dir / data_list, "r") as file_stream:
            for line in file_stream.readlines():
                img_path: str
                img_path, _ = line.split(",")
                data_pair[0].append(str(self._data_dir / "processed/oxford-pet/images" / img_path))
                data_pair[1].append(str(self._data_dir / "processed/oxford-pet/masks" / img_path))
        log.info("Dataset %s has %d samples.", data_list, len(data_pair[0]))  # noqa: WPS323
        data_pair = np.array(data_pair)
        if do_shuffle:
            data_pair = shuffle_data(data_pair)
        return data_pair[0], data_pair[1]

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
        if phase == PHASE_TRAIN:
            img_paths, mask_paths = self._img_train_paths, self._mask_train_paths
        elif phase == PHASE_VAL:
            img_paths, mask_paths = self._img_val_paths, self._mask_val_paths
        else:
            raise ValueError
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))

        dataset = dataset.map(
            load_data_pair_fn,
            num_parallel_calls=AUTOTUNE,
        ).prefetch(buffer_size=AUTOTUNE)

        dataset = dataset.cache()

        dataset = dataset.map(
            partial(
                process_data_fn,
                phase=phase,
                img_shape=self._img_shape,
                num_classes=self._n_classes,
            ),
            num_parallel_calls=AUTOTUNE,
        ).prefetch(buffer_size=AUTOTUNE)

        if phase == PHASE_TRAIN:
            dataset = dataset.shuffle(
                len(img_paths),
                reshuffle_each_iteration=True,
                seed=SHUFFEL_SEED,
            )
            dataset = dataset.repeat()

        dataset = dataset.batch(
            self._global_batch_size,
            drop_remainder=True,
        ).prefetch(buffer_size=AUTOTUNE)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        return dataset.with_options(options)
