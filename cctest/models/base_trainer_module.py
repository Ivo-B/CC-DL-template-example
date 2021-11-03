import platform

import hydra
import tensorflow as tf

from ..utils import utils

log = utils.get_logger(__name__)


class TrainingModule:
    def __init__(
        self,
        gpus: int,
        workers: int,
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

    def build(self):
        if platform.system() == "Windows":
            # Nccl does not work under windows
            cross_tower_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=self.gpus)
            strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_tower_ops)
        else:
            strategy = tf.distribute.MirroredStrategy()
        log.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        with strategy.scope():
            log.info(f"Instantiating model <{self.model['_target_']}>")
            self.model: tf.keras.Model = hydra.utils.instantiate(self.model)
            self.model.build()
            self.model.summary(print_fn=log.info)

            log.info(f"Instantiating loss <{self.loss_fn['_target_']}>")
            self.loss_fn = hydra.utils.instantiate(self.loss_fn)

            log.info(f"Instantiating metric <{self.metric_fn['_target_']}>")
            self.metric_fn = hydra.utils.instantiate(self.metric_fn)

            log.info(f"Instantiating lr scheduler <{self.lr_scheduler['_target_']}>")
            self.lr_scheduler = hydra.utils.instantiate(self.lr_scheduler)

            log.info(f"Instantiating optimizer <{self.optimizer_fn['_target_']}>")
            self.optimizer_fn = hydra.utils.instantiate(self.optimizer_fn)

            self.model.compile(
                optimizer=self.optimizer_fn,
                loss=self.loss_fn,
                metrics=self.metric_fn,
            )

    def fit(self, training_dataset, steps_per_epoch, validation_dataset=None):
        return self.model.fit(
            training_dataset,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_dataset,
            callbacks=self.callbacks + [self.lr_scheduler],
            workers=self.workers,
            max_queue_size=self.workers * 2,
            use_multiprocessing=True,
            shuffle=False,
        )

    def evaluate(self, test_dataset):
        return self.model.evaluate(test_dataset, use_multiprocessing=True)
