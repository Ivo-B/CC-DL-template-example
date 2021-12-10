import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss


class SegLoss(Loss):
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
        reduce_axis = list(np.arange(1, len(y_pred.shape) - 1))
        intersection = K.sum(y_true * y_pred, axis=reduce_axis)

        if self.squared_pred:
            y_true = K.pow(y_true, 2)
            y_pred = K.pow(y_pred, 2)

        ground_o = K.sum(y_true, axis=reduce_axis)
        pred_o = K.sum(y_pred, axis=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        loss_pro_class = (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr + 1e-8)
        if self.log_dice:
            loss_pro_class = -K.log(loss_pro_class)
        else:
            loss_pro_class = 1.0 - loss_pro_class

        # reducing only channel dimensions (not batch)
        return K.mean(loss_pro_class, axis=-1)

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:

        return self.get_seg_loss(y_true, y_pred)


class CESegLoss(SegLoss):
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

        f_loss: tf.Tensor = K.categorical_crossentropy(y_true, y_pred, from_logits=self.from_logits, axis=-1)
        # mean over all but batch axis
        mean_axes = [i for i in range(1, len(f_loss.shape))]
        f_loss = K.mean(f_loss, axis=mean_axes)

        return self.alpha * self.get_seg_loss(y_true, y_pred) + (1 - self.alpha) * f_loss


class BCESegLoss(SegLoss):
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
        f_loss: tf.Tensor = K.binary_crossentropy(y_true, y_pred, from_logits=self.from_logits)
        # mean over all but batch axis
        mean_axes = [i for i in range(1, len(f_loss.shape))]
        f_loss = K.mean(f_loss, axis=mean_axes)

        return self.alpha * self.get_seg_loss(y_true, y_pred) + (1 - self.alpha) * f_loss
