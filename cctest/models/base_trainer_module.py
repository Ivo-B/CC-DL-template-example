import tensorflow as tf
import hydra
import platform

import hydra
import keras.metrics
import tensorflow as tf

from cctest.utils import utils

log = utils.get_logger(__name__)


class TrainingModule:
    def __init__(
        self,
        gpus: int,
        epochs: int,
        optimizer: dict,
        lr_scheduler: dict,
        loss: dict,
        metric: dict,
        model: dict,
        callbacks: list[tf.keras.callbacks.Callback],
        logger: list[tf.keras.callbacks.Callback],
    ):
        self.gpus = gpus
        self.workers = workers
        self.epochs = epochs

        # combine all callbacks
        self.callbacks = callbacks + logger
        self.metric_fn = metric
        self.optimizer_fn = optimizer
        self.loss_fn = loss
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.model_config = model
        self.model: tf.keras.Model = None

    def build(self):
        if tf.config.list_logical_devices("GPU"):
            if platform.system() == "Windows":
                # Nccl does not work under windows
                cross_tower_ops = tf.distribute.ReductionToOneDevice()
                # strategy = tf.distribute.MultiWorkerMirroredStrategy(cross_device_ops=cross_tower_ops)
                strategy = tf.distribute.MultiWorkerMirroredStrategy()
            else:
                strategy = tf.distribute.MultiWorkerMirroredStrategy()
        else:  # Use the Default Strategy
            strategy = tf.distribute.get_strategy()
        log.info("Number of devices: {}".format(strategy.num_replicas_in_sync))

        with strategy.scope():
            log.info(f"Instantiating model <{self.model_config['_target_']}>")
            self.model: tf.keras.Model = hydra.utils.instantiate(self.model_config)
            self.model.build_graph().summary(print_fn=log.info, line_length=88)

            log.info(f"Instantiating loss <{self.loss_fn['_target_']}>")
            self.loss_fn = hydra.utils.instantiate(self.loss_fn)

            log.info(f"Instantiating metric <{self.metric_fn['_target_']}>")
            self.metric_fn = hydra.utils.instantiate(self.metric_fn)

            log.info(f"Instantiating lr scheduler <{self.lr_scheduler['_target_']}>")
            self.lr_scheduler = hydra.utils.instantiate(self.lr_scheduler)

            log.info(f"Instantiating optimizer <{self.optimizer_fn['_target_']}>")
            self.optimizer_fn = hydra.utils.instantiate(self.optimizer_fn)

        log.info(f"Instantiating lr scheduler <{self.lr_scheduler['_target_']}>")
        self.lr_scheduler = hydra.utils.instantiate(self.lr_scheduler)

        self.model.compile(optimizer=self.optimizer_fn, loss=self.loss_fn, metrics=self.metric_fn)

    def fit(self, training_dataset: tf.data.Dataset, steps_per_epoch: int, validation_dataset: tf.data.Dataset = None):
        return self.model.fit(
            training_dataset,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_dataset,
            callbacks=self.callbacks + [self.lr_scheduler],
            verbose=1,
            workers=1,
            use_multiprocessing=False,
            shuffle=False,
        )

    def evaluate(self, test_dataset):
        return self.model.evaluate(test_dataset, use_multiprocessing=False)
