import io

import matplotlib.pyplot as plt
import numpy as np
import seaborn_image as isns
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from cctest.utils import utils

log = utils.get_logger(__name__)


class ImageLogger(Callback):
    def __init__(self, log_dir, epoch_freq, num_images, sample_batch, phase):
        super().__init__()
        self.file_writer_image = tf.summary.create_file_writer(log_dir + f'/images_{phase}')
        self.test_images, self.test_masks = sample_batch

        self.epoch_freq = epoch_freq
        self.num_images = num_images

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def plot_image_grid(self, test_image, test_mask, test_pred, test_pred_raw_out):
        f, axs = plt.subplots(nrows=2, ncols=2, figsize=(4, 4))
        isns.imgplot(test_image, ax=axs[0, 0], robust=True, interpolation="nearest", origin="upper")
        axs[0, 0].set_title("Image")

        isns.imgplot(test_mask, ax=axs[0, 1], cmap="viridis", cbar=True, interpolation="nearest", origin="upper")
        axs[0, 1].set_title("Mask")

        isns.imgplot(test_pred, ax=axs[1, 0], cmap="viridis", cbar=True, interpolation="nearest", origin="upper")
        axs[1, 0].set_title("Prediction")

        isns.imgplot(test_pred_raw_out, ax=axs[1, 1], cmap="viridis", cbar=True, interpolation="nearest", origin="upper")
        axs[1, 1].set_title("Probability Class 1")

        plt.tight_layout()
        return f

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.epoch_freq) == 0:
            # Use the model to predict the values from the validation dataset.
            test_pred_raw = self.model(self.test_images, training=False).numpy()
            if self.num_images > test_pred_raw.shape[0]:
                self.num_images = test_pred_raw.shape[0]

            for i in range(self.num_images):
                test_image = self.test_images[i].numpy().copy()
                # norm to 0,1
                test_image = (test_image - np.min(test_image)) * (1.0 / (np.max(test_image) - np.min(test_image)))


                test_mask = np.argmax(self.test_masks[i].numpy().copy(), axis=-1)
                test_pred = np.argmax(test_pred_raw[i], axis=-1)
                test_pred_raw_out = test_pred_raw[i][..., 1]

                figure = self.plot_image_grid(test_image, test_mask, test_pred, test_pred_raw_out)
                grid_image = self.plot_to_image(figure)
                # Log figure as an image summary.
                with self.file_writer_image.as_default():
                    tf.summary.image(f"Example {i}", grid_image, step=epoch)


class LearningRateLogger(Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "learning_rate" in logs:
            return
        logs["learning_rate"] = self.model.optimizer.lr
