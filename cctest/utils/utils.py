import logging
import os
import random
import warnings
from typing import List, Sequence

import flatdict
import numpy as np
import rich.syntax
import rich.table
import rich.tree
import tensorflow as tf
import wandb
from typing import Union
from omegaconf import DictConfig, OmegaConf


def set_all_seeds(seed_value: Union[str, int] = "0xCAFFEE") -> None:
    # Set a seed value
    if isinstance(seed_value, int):
        seed_value = seed_value
    else:
        seed_value = int(seed_value, 0)
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    # for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
    #     setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger(__name__)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")


def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
        "name",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    # add log dir
    branch = tree.add("log_dir", style=style, guide_style=style)
    branch_content = os.getcwd()
    branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def print_history(
    history: dict,
) -> None:
    """Prints content of training history using Rich library and its table structure.
    Args:
        history (dict): Results from keras fit.
    """

    style = "dim"
    table = rich.table.Table(title="HISTORY", show_header=True, header_style="bold magenta")
    all_rows = []
    table.add_column("Epoch", style=style, justify="right")
    all_rows.append(["Epoch"])
    for idx, field in enumerate(history.keys()):
        idx += 1
        table.add_column(field, style=style, justify="right")
        all_rows.append([field])
        for epoch, entry in enumerate(history[field]):
            all_rows[idx].append(entry)
            if idx == 1:
                all_rows[0].append(epoch)

    # skipping header row
    for num_row in range(1, len(all_rows[0])):
        row = ()
        for num_col in range(len(all_rows)):
            if all_rows[num_col][0] == "lr":
                row += ("{:.4e}".format(all_rows[num_col][num_row]),)
            elif all_rows[num_col][0] == "Epoch":
                row += ("{:d}".format(all_rows[num_col][num_row]),)
            else:
                row += ("{:.4f}".format(all_rows[num_col][num_row]),)
        table.add_row(*row)

    rich.print(table)

    with open("history_table.txt", "w") as fp:
        rich.print(table, file=fp)


def empty(*args, **kwargs):
    pass


def log_hyperparameters(
    config: DictConfig,
    datamodule: "TfDataloader",
    trainer: "TrainingModule",
) -> None:
    """This method controls which parameters from Hydra config are saved to wandb.
    Additional saves:
        - number of trainable model parameters
    """
    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = OmegaConf.to_container(config["trainer"], resolve=True)
    hparams["model"] = OmegaConf.to_container(config["model"], resolve=True)
    hparams["datamodule"] = OmegaConf.to_container(config["datamodule"], resolve=True)
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = OmegaConf.to_container(config["callbacks"], resolve=True)

    # save number of model parameters
    hparams["model"]["params_total"] = trainer.model.count_params()
    hparams["model"]["params_trainable"] = np.sum([np.prod(v.get_shape()) for v in trainer.model.trainable_weights])
    hparams["model"]["params_not_trainable"] = np.sum([np.prod(v.get_shape()) for v in trainer.model.non_trainable_weights])

    # flatten nested dicts
    hparams = dict(flatdict.FlatDict(hparams, delimiter="/"))
    # send hparams to wandb logger
    wandb.config.update(hparams)


def finish(
    config: DictConfig,
    datamodule: "TfDataloader",
    trainer: "TrainingModule",
    callbacks: List[tf.keras.callbacks.Callback],
    logger: List[tf.keras.callbacks.Callback],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    wandb.finish()
