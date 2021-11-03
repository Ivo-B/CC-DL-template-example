import io
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seaborn_image as isns
import tensorflow as tf
from skimage.color import label2rgb
from tensorflow.keras.callbacks import Callback


class ImageLogger(Callback):
    def __init__(self, file_writer_image, batch_data, epoch_freq=1, num_images=16, testing=False):
        super().__init__()
        self.file_writer_image = file_writer_image
        self.test_images, self.test_masks = batch_data
        self.test_images = self.test_images.numpy()
        self.test_masks = self.test_masks.numpy()

        self.test_images_ds = tf.data.Dataset.from_tensors(self.test_images).cache()
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        self.test_images_ds = self.test_images_ds.with_options(options)

        self.epoch_freq = epoch_freq
        self.num_images = num_images
        self.testing = testing

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
                returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def plot_image_grid(self, test_image, test_mask, test_pred):
        # colors = sns.color_palette("colorblind", 3)
        # test_pred_rgb = label2rgb(test_pred, bg_label=0, colors=colors)
        # test_pred_over = label2rgb(test_pred, image=test_image, alpha=0.5, bg_label=0, colors=colors)
        # test_mask = label2rgb(test_mask, bg_label=0, colors=colors)

        f, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        isns.imgplot(test_image, ax=axs[0], cmap='gray', cbar=False, interpolation='nearest', origin='upper')
        isns.imgplot(test_mask, ax=axs[1], cmap='viridis', cbar=False, interpolation='nearest', origin='upper')
        isns.imgplot(test_pred, ax=axs[2], cmap='viridis', cbar=False, interpolation='nearest', origin='upper')
        # isns.imgplot(test_pred_over, ax=axs[3], cbar=False, interpolation='nearest')
        plt.tight_layout()
        if self.testing:
            plt.show()
        return f

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.epoch_freq) == 0:
            # Use the model to predict the values from the validation dataset.
            # test_pred_raw = self.model(self.test_images, training=False)
            test_pred_raw = self.model.predict(self.test_images_ds)
            test_pred_raw = tf.nn.softmax(test_pred_raw, axis=-1)
            if self.num_images > test_pred_raw.shape[0]:
                self.num_images = test_pred_raw.shape[0]

            for i in range(self.num_images):
                test_image = self.test_images[i].copy()
                test_image += 0.5
                test_image = np.concatenate((test_image, test_image, test_image), axis=-1)
                test_mask = np.argmax(self.test_masks[i].copy(), axis=-1)
                test_pred = np.argmax(test_pred_raw[i], axis=-1)

                figure = self.plot_image_grid(test_image, test_mask, test_pred)
                grid_image = self.plot_to_image(figure)
                # Log the confusion matrix as an image summary.
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
