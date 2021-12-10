from typing import List, Optional

import hydra
import tensorflow as tf
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tensorflow.keras.callbacks import Callback
from {{cookiecutter.module_name}}.datamodule.base_datamodule import TfDataloader
from {{cookiecutter.module_name}}.model.base_model_trainer import TrainingModule
from {{cookiecutter.module_name}}.utils import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all objects from config.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    using_wandb = False

    # Set seed for random number generators in tensorflow, numpy and python.random
    if config.get("seed"):
        utils.set_all_seeds(config.seed)

    # check devices
    physical_devices = tf.config.list_physical_devices("GPU")
    if config.trainer.get("gpus") >= 1 and config.trainer.get("gpus") > len(physical_devices):
        raise ValueError(f"Did not find enough GPUs. Reduce the number of GPUs to use!")
    if config.trainer.get("gpus") == 0 and len(physical_devices) > 0:
        # hiding all GPUs!
        tf.config.set_visible_devices([], "GPU")
    elif 0 < config.trainer.get("gpus") < len(physical_devices):
        gpus_to_hide = len(physical_devices) - config.trainer.get("gpus")
        # hiding some GPUs!
        tf.config.set_visible_devices(physical_devices[gpus_to_hide:], "GPU")
    else:
        # use all GPUs
        config.trainer.gpus = len(physical_devices)
    # TF usually allocates all memory of the GPU
    if config.trainer.get("gpus") > 0:
        visible_devices = tf.config.get_visible_devices("GPU")
        for gpu in visible_devices:
            tf.config.experimental.set_memory_growth(gpu, True)

    #############################
    # Doing data stuff
    #############################
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: TfDataloader = hydra.utils.instantiate(
        config.datamodule,
        num_gpus=config.trainer.get("gpus"),
        data_aug=config.datamodule.get("data_aug"),
        _convert_=None,
        _recursive_=False,
    )
    training_dataset = datamodule.get_tf_dataset("training")
    validation_dataset = datamodule.get_tf_dataset("validation")

    #########################
    # Callbacks for model training
    #########################
    callbacks: List[Callback] = []
    if "callback" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init loggers
    logger: List[Callback] = []
    if "logger" in config:
        for lg_key, lg_conf in config.logger.items():
            if "image_logger" in lg_key:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(
                    hydra.utils.instantiate(
                        lg_conf,
                        sample_batch=next(iter(training_dataset)),
                        phase="train",
                        _recursive_=False,
                    ),
                )
                logger.append(
                    hydra.utils.instantiate(
                        lg_conf,
                        sample_batch=next(iter(validation_dataset)),
                        phase="val",
                        _recursive_=False,
                    ),
                )
                continue

            if "wandb_init" in lg_key:
                using_wandb = True
                log.info(
                    f"Instantiating wandb run: <entity={lg_conf.user}>, <project={lg_conf.project}>, <name={lg_conf.name}>",
                )
                wandb.init(entity=lg_conf.user, project=lg_conf.project, name=lg_conf.name)

            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    #########################
    # Initialize model and trainer
    #########################
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: TrainingModule = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks + logger,  # CB and Logger are combined for TF
        model=config.model,
        _convert_=None,
        _recursive_=False,
    )
    trainer.build()

    # Send some parameters from config to wandb logger
    if using_wandb:
        log.info("Logging hyperparameters to wandb!")
        utils.log_hyperparameters(
            config=config,
            datamodule=datamodule,
            trainer=trainer,
        )

    # Train the model
    log.info("Starting training!")

    history = trainer.fit(
        training_dataset,
        steps_per_epoch=datamodule.steps_per_epoch,
        validation_dataset=validation_dataset,
        verbose=1 if config.get("debug_mode") else 2,
    )

    if config.get("print_history"):
        utils.print_history(history.history, config.trainer.validation_freq)

    # Make sure everything closed properly
    if using_wandb:
        # without this sweeps with wandb logger might crash!
        wandb.finish()

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        if HydraConfig.get().sweeper.get("direction") == "maximize":
            return max(history.history[optimized_metric])
        elif HydraConfig.get().sweeper.get("direction") == "minimize":
            return min(history.history[optimized_metric])
        else:
            raise ValueError
