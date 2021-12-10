import hydra
import tensorflow as tf
from omegaconf import DictConfig
from {{cookiecutter.module_name}}.utils import utils

log = utils.get_logger(__name__)


class TrainingModule:
    def __init__(
        self,
        gpus: int,
        epochs: int,
        validation_freq: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        loss: DictConfig,
        metric: DictConfig,
        model: DictConfig,
        callbacks: list[tf.keras.callbacks.Callback],
    ):
        self.gpus = gpus
        self.epochs = epochs
        self.validation_freq = validation_freq

        self.callbacks = list(callbacks)
        self.metric_config = metric
        self.optimizer_config = optimizer
        self.loss_config = loss
        self.lr_scheduler_config = lr_scheduler
        self.model_config = model
        self.model: tf.keras.Model = None

    def build(self):
        if tf.config.list_logical_devices("GPU"):
            strategy = tf.distribute.MirroredStrategy()
        else:  # Use the Default Strategy
            strategy = tf.distribute.get_strategy()
        log.info("Number of devices: {}".format(strategy.num_replicas_in_sync))

        with strategy.scope():
            log.info(f"Instantiating model <{self.model_config._target_}>")
            self.model: tf.keras.Model = hydra.utils.instantiate(self.model_config)
            self.model.build_graph().summary(print_fn=log.info, line_length=88)

            log.info(f"Instantiating loss <{self.loss_config._target_}>")
            loss_fn = hydra.utils.instantiate(self.loss_config)

            metric_fn = []
            for _, metric_conf in self.metric_config.items():
                if "_target_" in metric_conf:
                    log.info(f"Instantiating metric <{metric_conf._target_}>")
                    metric_fn.append(hydra.utils.instantiate(metric_conf))

        log.info(f"Instantiating lr scheduler <{self.lr_scheduler_config._target_}>")
        self.callbacks.append(hydra.utils.instantiate(self.lr_scheduler_config))

        log.info(f"Instantiating optimizer <{self.optimizer_config._target_}>")
        optimizer_fn = hydra.utils.instantiate(self.optimizer_config)

        self.model.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=metric_fn)

    def fit(
        self,
        training_dataset: tf.data.Dataset,
        steps_per_epoch: int,
        validation_dataset: tf.data.Dataset = None,
        verbose: int = 2,
    ):
        return self.model.fit(
            training_dataset,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_dataset,
            validation_freq=self.validation_freq,
            callbacks=self.callbacks,
            verbose=verbose,
            workers=1,
            use_multiprocessing=False,
            shuffle=False,
        )

    def evaluate(self, test_dataset):
        return self.model.evaluate(test_dataset, use_multiprocessing=False)
