import io

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from {{cookiecutter.module_name}}.utils import utils

log = utils.get_logger(__name__)


class ImageLogger(Callback):
    def __init__(self, log_dir, epoch_freq, num_images, sample_batch, phase):
        super().__init__()
        self.file_writer_image = tf.summary.create_file_writer(log_dir + f"/images_{phase}")
        # note model(test_data) does not work well with model.fit()
        self.test_images, self.test_masks = sample_batch
        self.test_images = self.test_images.numpy()
        self.test_masks = self.test_masks.numpy()

        self.test_images_ds = tf.data.Dataset.from_tensors(self.test_images).cache()
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        self.test_images_ds = self.test_images_ds.with_options(options)

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
        axs[0, 0].imshow(test_image, interpolation="nearest")
        axs[0, 0].set_title("Image")

        axs[0, 1].imshow(test_mask, cmap="viridis", interpolation="nearest")
        axs[0, 1].set_title("Mask")

        axs[1, 0].imshow(test_pred, cmap="viridis", interpolation="nearest")
        axs[1, 0].set_title("Prediction")

        axs[1, 1].imshow(test_pred_raw_out, cmap="viridis", interpolation="nearest")
        axs[1, 1].set_title("Probability Class 1")

        plt.tight_layout()
        return f

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.epoch_freq) == 0:
            test_pred_raw = self.model.predict(self.test_images_ds)

            if self.num_images > test_pred_raw.shape[0]:
                self.num_images = test_pred_raw.shape[0]

            for i in range(self.num_images):
                test_image = self.test_images[i].copy()
                # norm to 0,1
                test_image = (test_image - np.min(test_image)) * (1.0 / (np.max(test_image) - np.min(test_image)))

                test_mask = np.argmax(self.test_masks[i], axis=-1)
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
