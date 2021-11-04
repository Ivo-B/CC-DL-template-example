"""MNIST handwritten digits dataset.
This is a dataset of 60,000 28x28 grayscale images of the 10 digits,
along with a test set of 10,000 images.
More info can be found at the
[MNIST homepage](http://yann.lecun.com/exdb/mnist/).

License:
    Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset,
    which is a derivative work from original NIST datasets.
    MNIST dataset is made available under the terms of the
    [Creative Commons Attribution-Share Alike 3.0 license.](
    https://creativecommons.org/licenses/by-sa/3.0/)
"""
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
from keras.utils.data_utils import get_file

# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())
path = Path(os.environ.get('PROJECT_DIR')) / 'data' / 'raw'

origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
path_download = get_file(
  path / 'mnist.npz',
  origin=origin_folder + 'mnist.npz',
  file_hash=
  '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1')


with np.load(path_download, allow_pickle=True) as f:  # pylint: disable=unexpected-keyword-arg
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

all_x = np.concatenate((x_train, x_test))
all_y = np.concatenate((y_train, y_test))


all_x = all_x/255


path = Path(os.environ.get('PROJECT_DIR')) / 'data' / 'processed'

for idx in range(len(all_x)):
    np.save(path / f'{idx:05d}', all_x[idx,...])


from sklearn.model_selection import train_test_split
X_train_val, X_test, y_train_val, y_test = train_test_split(all_x, all_y, test_size=10000, random_state=42, shuffle=False)


path = Path(os.environ.get('PROJECT_DIR')) / 'data'

with open(path / 'test_data.txt', 'a') as f:
    for name, x, y in zip(list(range(60000, 70000)), X_test, y_test):
        assert np.all(all_x[name] == x) == True
        f.write(f"{name:05d}.npy, {y}\n")

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=5000, random_state=42, shuffle=False)

with open(path / 'training_data.txt', 'a') as f:
    for name, x, y in zip(list(range(0, 55000)), X_train, y_train):
        assert np.all(all_x[name] == x) == True
        f.write(f"{name:05d}.npy, {y}\n")


with open(path / 'validation_data.txt', 'a') as f:
    for name, x, y in zip(list(range(55000, 60000)), X_val, y_val):
        assert np.all(all_x[name] == x) == True
        f.write(f"{name:05d}.npy, {y}\n")
