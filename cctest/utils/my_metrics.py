import warnings

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
from tensorflow.python.ops import (
    array_ops,
    confusion_matrix,
    init_ops,
    math_ops,
)


class MeanSegMetric(Metric):
    """Computes the mean Dice metric. """

    def __init__(self, jaccard, from_logits, include_background, num_classes, dtype="float32"):
        if jaccard:
            super(MeanSegMetric, self).__init__(name="mean_iou", dtype=dtype)
        else:
            super(MeanSegMetric, self).__init__(name="mean_dice", dtype=dtype)
        self.jaccard = jaccard
        self.num_classes = num_classes
        self.from_logits = from_logits
        self.include_background = include_background

        # Variable to accumulate the predictions in the confusion matrix.
        self.total_cm = self.add_weight(
            "total_confusion_matrix", shape=(num_classes, num_classes), initializer=init_ops.zeros_initializer
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.

        Returns:
          Update op.
        """

        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)

        if self.from_logits:
            y_pred = K.softmax(y_pred, axis=-1)

        # get one-hot encoded masks from y_pred (true mask should already be one-hot)
        y_pred = K.one_hot(K.argmax(y_pred), self.num_classes)

        if not self.include_background:
            n_pred_ch = y_pred.shape[-1]
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                y_true = y_true[..., 1:]
                y_pred = y_pred[..., 1:]

        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = array_ops.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = array_ops.reshape(y_true, [-1])

        if sample_weight is not None:
            sample_weight = math_ops.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = array_ops.reshape(sample_weight, [-1])

        # Accumulate the prediction to current confusion matrix.
        current_cm = confusion_matrix.confusion_matrix(
            y_true, y_pred, self.num_classes, weights=sample_weight, dtype=self._dtype
        )
        return self.total_cm.assign_add(current_cm)

    def result(self):
        """Compute the mean dice/iou via the confusion matrix.

        DICE: 2 * true positives / 2 * true positives + false positives + false negatives
        IoU: true positives / true positives + false positives + false negatives
        """
        sum_over_row = math_ops.cast(math_ops.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = math_ops.cast(math_ops.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = math_ops.cast(array_ops.tensor_diag_part(self.total_cm), dtype=self._dtype)
        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        if self.jaccard:
            denominator = sum_over_row + sum_over_col - true_positives
        else:
            denominator = sum_over_row + sum_over_col

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = math_ops.reduce_sum(math_ops.cast(math_ops.not_equal(denominator, 0), dtype=self._dtype))

        if self.jaccard:
            metric = math_ops.div_no_nan(true_positives, denominator)
        else:
            metric = math_ops.div_no_nan(2 * true_positives, denominator)

        return math_ops.div_no_nan(math_ops.reduce_sum(metric), num_valid_entries)

    def reset_state(self):
        K.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        config = {"num_classes": self.num_classes, "jaccard": self.jaccard}
        base_config = super(MeanSegMetric, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
