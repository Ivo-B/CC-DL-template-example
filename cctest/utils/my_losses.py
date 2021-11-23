import warnings
from typing import Callable, Optional

import keras.losses
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss


class SegLoss(Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    Input logits `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).
    Axis N of `input` is expected to have logit predictions for each class rather than being image channels,
    while the same axis of `target` can be 1 or N (one-hot format). The `smooth_nr` and `smooth_dr` parameters are
    values added to the intersection and union components of the inter-over-union calculation to smooth results
    respectively, these values should be small. The `include_background` class attribute can be set to False for
    an instance of DiceLoss to exclude the first category (channel index 0) which is by convention assumed to be
    background. If the non-background segmentations are small compared to the total image size they can get
    overwhelmed by the signal from the background so excluding it in such cases helps convergence.

    Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric Medical Image Segmentation, 3DV, 2016.

    """

    def __init__(
        self,
        from_logits: bool = False,
        include_background: bool = True,
        squared_pred: bool = False,
        jaccard: bool = False,
        log_dice: bool = False,
        smooth_nr: float = 1.0,
        smooth_dr: float = 1.0,
        dtype: str = "float32",
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.

        Raises:

        """
        if jaccard:
            super(SegLoss, self).__init__(name="IouLoss")
        elif log_dice:
            super(SegLoss, self).__init__(name="LogDiceLoss")
        else:
            super(SegLoss, self).__init__(name="DiceLoss")

        self.from_logits = from_logits
        self.include_background = include_background
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.log_dice = log_dice
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.dtype = dtype

    def get_seg_loss(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Args:
            input: the shape should be BH[WD]N, where N is the number of classes.
            target: the shape should be BH[WD]N, where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if setted)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        y_true = tf.cast(y_true, dtype=self.dtype)
        y_pred = tf.cast(y_pred, dtype=self.dtype)

        n_pred_ch = y_pred.shape[-1]
        if self.from_logits:
            y_pred = K.softmax(y_pred, axis=-1)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                y_true = y_true[..., 1:]
                y_pred = y_pred[..., 1:]

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = list(np.arange(1, len(y_pred.shape) - 1))
        intersection = K.sum(y_true * y_pred, axis=reduce_axis)

        if self.squared_pred:
            y_true = K.pow(y_true, 2)
            y_pred = K.pow(y_pred, 2)

        ground_o = K.sum(y_true, axis=reduce_axis)
        pred_o = K.sum(y_pred, axis=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: tf.Tensor = (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr + 1e-8)
        if self.log_dice:
            f = -K.log(f)
        else:
            f: tf.Tensor = 1.0 - f

        tf.debugging.check_numerics(f, "test 123")

        # reducing only channel dimensions (not batch)
        return K.mean(f, axis=-1)

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Args:
            input: the shape should be BH[WD]N, where N is the number of classes.
            target: the shape should be BH[WD]N, where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if setted)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        return self.get_seg_loss(y_true, y_pred)


class CESegLoss(SegLoss):
    """"""
    def __init__(
        self,
        from_logits: bool = True,
        alpha: float = 0.5,
        include_background: bool = True,
        squared_pred: bool = False,
        jaccard: bool = False,
        log_dice: bool = False,
        smooth_nr: float = 1.0,
        smooth_dr: float = 1.0,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.

        Raises:

        """
        super().__init__(from_logits=from_logits, jaccard=jaccard, log_dice=log_dice)
        self.alpha = float(alpha)
        self.include_background = include_background
        self.squared_pred = squared_pred
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Args:
            input: the shape should be BH[WD]N, where N is the number of classes.
            target: the shape should be BH[WD]N, where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if setted)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        f_loss: tf.Tensor = K.categorical_crossentropy(y_true, y_pred, from_logits=self.from_logits, axis=-1)
        # mean over all but batch axis
        mean_axes = [i for i in range(1, len(f_loss.shape))]
        f_loss = K.mean(f_loss, axis=mean_axes)

        return self.alpha * self.get_seg_loss(y_true, y_pred) + (1 - self.alpha ) * f_loss


class BCESegLoss(SegLoss):
    """"""

    def __init__(
        self,
        from_logits: bool = True,
        alpha: float = 0.5,
        include_background: bool = True,
        squared_pred: bool = False,
        jaccard: bool = False,
        log_dice: bool = False,
        smooth_nr: float = 1.0,
        smooth_dr: float = 1.0,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.

        Raises:

        """
        super().__init__(from_logits=from_logits, jaccard=jaccard, log_dice=log_dice)
        self.include_background = include_background
        self.alpha = float(alpha)
        self.squared_pred = squared_pred
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Args:
            input: the shape should be BH[WD]N, where N is the number of classes.
            target: the shape should be BH[WD]N, where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if setted)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        f_loss: tf.Tensor = K.binary_crossentropy(y_true, y_pred, from_logits=self.from_logits)
        # mean over all but batch axis
        mean_axes = [i for i in range(1, len(f_loss.shape))]
        f_loss = K.mean(f_loss, axis=mean_axes)

        return self.alpha * self.get_seg_loss(y_true, y_pred) +  (1 - self.alpha ) * f_loss
