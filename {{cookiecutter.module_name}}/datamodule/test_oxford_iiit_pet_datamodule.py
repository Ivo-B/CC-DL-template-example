import os
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from dotenv import find_dotenv, load_dotenv

from {{ cookiecutter.module_name }}.datamodule.oxford_iiit_pet_datamodule import OxfordPetDataset

load_dotenv(find_dotenv())

path = Path(os.environ.get("PROJECT_DIR")) / "data"

datamodule = OxfordPetDataset(
    str(path),
    "OxfordPet_training_data.txt",
    "OxfordPet_validation_data.txt",
    "OxfordPet_test_data.txt",
    32,
    1,
)

train_sample = next(iter(datamodule.get_tf_dataset("training")))

for i in range(32):
    img = train_sample[0][i]
    mask = tf.argmax(train_sample[1][i], axis=-1)

    img = img.numpy()
    img = (img * (0.229, 0.224, 0.225)) + (0.485, 0.456, 0.406)
    plt.imshow(img)
    plt.show()
    plt.imshow(mask)
    plt.show()
