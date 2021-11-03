# Our example cookiecutter project
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9-3670A0?style=flat-square&logo=python&logoColor=ffdd54"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="Tensorflow" src="https://img.shields.io/badge/-Tensorflow 2.4-%23FF6F00?style=flat-square&logo=Tensorflow&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=flat-square&labelColor=gray"></a>
<a href="https://hub.docker.com/r/ashlev/lightning-hydra"><img alt="Docker" src="https://img.shields.io/badge/docker-257bd6?style=flat-square&logo=docker&logoColor=white"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg?style=flat-square)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg?style=flat-square)](https://papers.nips.cc/paper/2020)

## Description
We need some project explanation here!

## Requirements to use this project
- poetry
- python 3.9

Install pyenv and poetry:
````yaml
# PyEnv for Ubuntu
curl https://pyenv.run | bash

# Poetry for Ubuntu
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
````

## How to use
````yaml
# clone project
git clone https://github.com/Ivo-B/Example_CC_DL_template
cd Example_CC_DL_template

poetry install
# use poetry
poetry shell
# or pyenv
source ./.venv/Scripts/activate

#activate pre-commit
pre-commit install
````
Template contains example with MNIST classification.<br>
 1. edit [.env.example](.env.example) and set your PROJECT_PATH, rename file to `.env`
 2. run jupyter notebook `notebooks/1.0-IBa-data-download-and-processed.ipynb` to download and prepare data
 3. Now your can simply run `python run.py`.


## Project Organization
```
├──.venv                <- Local poetry environment
│   └──.gitkeep
├── configs                 <- Hydra configuration files
│   ├── callbacks               <- Callbacks configs
│   ├── datamodule              <- Datamodule configs
│   ├── experiment              <- Experiment configs
│   ├── hparams_search          <- Hyperparameter search configs
│   ├── mode                    <- Running mode configs
│   ├── logger                  <- Logger configs
│   ├── model                   <- Model configs
│   ├── trainer                 <- Trainer configs
│   │   ├── loss                <- TODO
│   │   ├── lr_scheduler        <- TODO
│   │   ├── metric              <- TODO
│   │   └── optimizer           <- TODO
│   │
│   └── config.yaml             <- Main project configuration file
├── data
│   ├── external        <- Data from third party sources.
│   ├── interim         <- Intermediate data that has been transformed.
│   ├── processed       <- The final, canonical data sets for modeling.
│   └── raw             <- The original, immutable data dump.
├── docs                <- A default Sphinx project; see sphinx-doc.org for details
├── models              <- Trained and serialized models, model predictions, or model summaries
├── notebooks           <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
├── references          <- Data dictionaries, manuals, and all other explanatory materials.
├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures         <- Generated graphics and figures to be used in reporting
├── test                <- Data dictionaries, manuals, and all other explanatory materials.
├── cctest              <- Source code for use in this project.
│   │
│   ├── data                              <- Scripts to download or generate data
│   │   └── __init__.py
│   ├── dataloaders                       <- Scripts to handel and load the preprocessed data
│   │   ├── __init__.py
│   │   ├── base_datamodule.py            <- TODO
│   │   └── mnist_datamodule.py           <- TODO
│   ├── evaluation                        <- Scripts to do evaluation of the results
│   │   └── __init__.py
│   ├── executor                          <- Scripts to train, eval and test models
│   │   ├── __init__.py
│   │   └── train_model.py                <- TODO
│   ├── models                            <- Scripts to define model architecture
│   │   ├── modules                       <- TODO
│   │   │   ├── __init__.py
│   │   │   ├── simple_conv_net.py        <- TODO
│   │   │   └── simple_dense_net.py       <- TODO
│   │   ├── __init__.py
│   │   └── base_trainer_module.py        <- TODO
│   ├── utils                             <- Utility scripts
│   │   ├── __init__.py
│   │   ├── my_callback.py                <- TODO
│   │   └── utils.py                      <- TODO
│   │
│   ├── visualization                     <- Scripts to create exploratory and results oriented
│   │   └── __init__.py                    visualizations
│   │
│   └── __init__.py                       <- Makes cctest a Python module
│
├── .editorconfig         <- file with format specification. You need to install
│                             the required plugin for your IDE in order to enable it.
├── .gitignore         <- file that specifies what should we commit into
│                             the repository and we should not.
├── LICENSE
├── poetry.toml         <- poetry config file to install enviroment locally
├── poetry.lock         <- lock file for dependencies. It is used to install exactly
│                         the same versions of dependencies on each build
├── pyproject.toml      <- The project's dependencies for reproducing the
│                         analysis environment
├── README.md           <- The top-level README for developers using this project.
└── setup.cfg           <- configuration file, that is used by all tools in this project```
```

## Guide

### How To Get Started

- First, you should probably get familiar with [PyTorch Lightning](https://www.pytorchlightning.ai)
- Next, go through [Hydra quick start guide](https://hydra.cc/docs/intro/) and [basic Hydra tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/)
  <br>

### How it works

By design, every run is initialized by [run_training.py](run_training.py) file. All modules are dynamically instantiated from module paths specified in config. Example model config:

```yaml
_target_: cctest.models.modules.simple_dense_net.SimpleDenseNet
input_shape: [28,28,1]
lin1_size: 256
lin2_size: 256
lin3_size: 256
output_size: 10
```

Using this config we can instantiate the object with the following line:

```python
model = hydra.utils.instantiate(config.model)
```

This allows you to easily iterate over new models!<br>
Every time you create a new one, just specify its module path and parameters in appriopriate config file. <br>
The whole pipeline managing the instantiation logic is placed in [cctest/executor/train_model.py](cctest/executor/train_model.py).

<br>

### Main Project Configuration

Location: [configs/config.yaml](configs/config.yaml)<br>
Main project config contains default training configuration.<br>
It determines how config is composed when simply executing command `python run.py`.<br>
It also specifies everything that shouldn't be managed by experiment configurations.

<details>
<summary><b>Show main project configuration</b></summary>

```yaml
# specify here default training configuration
defaults:
  - trainer: default.yaml
  - model: mnist_model.yaml
  - datamodule: mnist_datamodule.yaml
  - callbacks: default.yaml # set this to null if you don't want to use callbacks
  - logger: null # set logger here or use command line (e.g. `python run.py logger=wandb`)

  - mode: default.yaml

  - experiment: null
  - hparams_search: null

# path to original working directory
# hydra hijacks working directory by changing it to the current log directory,
# so it's useful to have this path as a special variable
# learn more here: https://hydra.cc/docs/next/tutorials/basic/running_your_app/working_directory
work_dir: ${hydra:runtime.cwd}

# path to folder with data
data_dir: ${work_dir}/data/

# pretty print config at the start of the run using Rich library
print_config: True

# disable python warnings if they annoy you
ignore_warnings: True
```

</details>
<br>

### Experiment Configuration

Location: [configs/experiment](configs/experiment)<br>
You should store all your experiment configurations in this folder.<br>
Experiment configurations allow you to overwrite parameters from main project configuration.

**Simple example**

```yaml
# @package _global_

# to execute this experiment run:
# python run.py experiment=example_simple.yaml

defaults:
  - override /trainer: default.yaml
  - override /model: mnist_model_conv.yaml
  - override /datamodule: mnist_datamodule.yaml
  - override /callbacks: default.yaml
  - override /logger: null

# all parameters below will be merged with parameters from default configurations set above
# this allows you to overwrite only specified parameters

seed: 12345

trainer:
  epochs: 5

model:
  conv1_size: 32
  conv2_size: 64
  conv3_size: 128
  conv4_size: 256
  output_size: 10

datamodule:
  batch_size: 64
```

</details>

<details>
<summary><b>Show advanced example</b></summary>

```yaml
# @package _global_

# to execute this experiment run:
# python run.py experiment=example_full.yaml

defaults:
  - override /trainer: null # override trainer to null so it's not loaded from main config defaults...
  - override /model: null
  - override /datamodule: null
  - override /callbacks: null
  - override /logger: null

# we override default configurations with nulls to prevent them from loading at all
# instead we define all modules and their paths directly in this config,
# so everything is stored in one place

seed: 12345

trainer:
  loss:
    _target_: tensorflow.keras.losses.CategoricalCrossentropy
    from_logits: True
  metric:
    _target_: tensorflow.keras.metrics.CategoricalAccuracy
  optimizer:
    _target_: tensorflow.keras.callbacks.ReduceLROnPlateau
    factor: 0.5
    patience: 10
    min_lr: 1e-9
    verbose: 1
    monitor: 'val_loss'
    mode: 'min'
  lr_scheduler:
    _target_: tensorflow.keras.optimizers.Adam
    learning_rate: 0.0003
    beta_1: 0.9
    beta_2: 0.999
    epsilon: 1e-07
    amsgrad: 'false'

  _target_: cctest.models.base_trainer_module.TrainingModule
  # set `-1` to train on all GPUs in a node,
  # '>0' to train on specific num of GPUs in a node,
  # `0` to train on CPU only
  gpus: -1
  workers: 10
  epochs: 5
  # resume_from_checkpoint: ${work_dir}/last.ckpt

model:
  _target_: cctest.models.modules.simple_conv_net.SimpleConvNet
  input_shape: [28, 28, 1]
  conv1_size: 16
  conv2_size: 32
  conv3_size: 32
  conv4_size: 64
  output_size: 10

datamodule:
  _target_: cctest.datamodule.mnist_datamodule.MNISTDataset
  data_dir: ${data_dir} # data_dir is specified in config.yaml
  data_training_list: 'training_data.txt'
  data_val_list: 'validation_data.txt'
  data_test_list: 'test_data.txt'
  batch_size: 64

callbacks:
  model_checkpoint:
    _target_: tensorflow.keras.callbacks.ModelCheckpoint
    monitor: 'val_loss' # name of the logged metric which determines when model is improving
    mode: 'min' # can be 'max' or 'min'
    save_best_only: True # save best model (determined by above metric)
    save_freq: 'epoch' # 'epoch' or integer. When using 'epoch', the callback saves the model after each epoch. When using integer, the callback saves the model at end of this many batches.
    verbose: 0
    filepath: 'checkpoints/epoch_{epoch:03d}-{val_loss:.2f}.tf'
    save_format: 'tf'
  early_stopping:
    _target_: tensorflow.keras.callbacks.EarlyStopping
    monitor: 'val_loss' # name of the logged metric which determines when model is improving
    mode: 'max' # can be 'max' or 'min'
    patience: 100 # how many validation epochs of not improving until training stops
    min_delta: 0 # minimum change in the monitored metric needed to qualify as an improvement

logger:
  tensorboard:
    _target_: tensorflow.keras.callbacks.TensorBoard
    log_dir: "tensorboard/${name}"
    write_graph: False
    profile_batch: 0
  csv:
    _target_: tensorflow.keras.callbacks.CSVLogger
    filename: "./csv/${name}.csv"
```

</details>

<br>

### Workflow

1. Write your model (see [simple_conv_net.py](cctest/models/modules/simple_conv_net.py) for example)
2. Write your datamodule (see [mnist_datamodule.py](cctest/datamodules/mnist_datamodule.py) for example)
3. Write your experiment config, containing paths to your model and datamodule
4. Run training with chosen experiment config: `python run.py experiment=experiment_name`
   <br>

### Logs

Hydra creates new working directory for every executed run. <br>
By default, logs have the following structure:

```
│
├── logs
│   ├── runs                    # Folder for logs generated from single runs
│   │   ├── 2021-02-15              # Date of executing run
│   │   │   ├── 16-50-49                # Hour of executing run
│   │   │   │   ├── .hydra                  # Hydra logs
│   │   │   │   ├── wandb                   # Weights&Biases logs
│   │   │   │   ├── checkpoints             # Training checkpoints
│   │   │   │   └── ...                     # Any other thing saved during training
│   │   │   ├── ...
│   │   │   └── ...
│   │   ├── ...
│   │   └── ...
│   │
│   └── multiruns               # Folder for logs generated from multiruns (sweeps)
│       ├── 2021-02-15_16-50-49     # Date and hour of executing sweep
│       │   ├── 0                       # Job number
│       │   │   ├── .hydra                  # Hydra logs
│       │   │   ├── wandb                   # Weights&Biases logs
│       │   │   ├── checkpoints             # Training checkpoints
│       │   │   └── ...                     # Any other thing saved during training
│       │   ├── 1
│       │   ├── 2
│       │   └── ...
│       ├── ...
│       └── ...
│
```
