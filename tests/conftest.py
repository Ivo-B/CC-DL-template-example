import pytest
from omegaconf import DictConfig

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
        "max_pixel_value": 1.0
    },
})


@pytest.fixture()
def data_aug_oxfordpet() -> DictConfig:
    """Creates default prompt values."""
    DATA_AUG["norm_data"]["mean"] = [0.5, 0.5, 0.5]
    DATA_AUG["norm_data"]["std"] = [1, 1, 1]
    return DATA_AUG

@pytest.fixture()
def data_aug_mnist() -> DictConfig:
    """Creates default prompt values."""
    DATA_AUG["norm_data"]["mean"] = [0.1307]
    DATA_AUG["norm_data"]["std"] = [0.3081]
    return DATA_AUG
