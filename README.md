# Welcome to {{cookiecutter.project_name}}
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9-3670A0?style=flat-square&logo=python&logoColor=ffdd54"></a>
{% if cookiecutter.dl_framework == "Tensorflow" %}
<a href="https://www.tensorflow.org/install/"><img alt="Tensorflow" src="https://img.shields.io/badge/-Tensorflow 2.7-%23FF6F00?style=flat-square&logo=Tensorflow&logoColor=white"></a>
{% elif cookiecutter.ml_framework == "PyTorch" %}
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.10-ee4c2c?style=flat-square&logo=pytorch&logoColor=white"></a>
{% endif %}
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=flat-square&labelColor=gray"></a>
<a href="https://www.docker.com/"><img alt="Docker" src="https://img.shields.io/badge/docker-257bd6?style=flat-square&logo=docker&logoColor=white"></a>
<a href="https://github.com/psf/black"><img alt="Black" src="https://img.shields.io/badge/code%20style-black-black?style=flat-square"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg?style=flat-square)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg?style=flat-square)](https://papers.nips.cc/paper/2020)

## Description
{{cookiecutter.description}}

## Requirements to use this project
- poetry
- python 3.9

You can use your favorite method to provide python 3.9 for poetry. We recommend [https://github.com/pyenv/pyenv#installation](PyEnv), but you can also use [https://docs.conda.io/en/latest/miniconda.html](Conda) etc.

Install poetry:
````yaml
# Poetry for Ubuntu
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
````

## How to use
````yaml
# clone project
git clone https://repository-provider/username/{{cookiecutter.repo_name}}
cd {{cookiecutter.repo_name}}
````

If you use conda to provide the correct Python version, installation and usability is a bit different:
````yaml
# creates a conda environment
make environment
# run to activate env, install packages
source ./bash/finalize_environment.sh
````

When you use PyEnv to provide Python, your virtualenvironment is installed in the project folder `.venv`
````yaml
poetry install
# activate Virtualenv by
source ./.venv/Scripts/activate
````

Template contains examples with MNIST classification and Oxfordpet segmentation.<br>
 1. edit [.env.example](.env.example) and set your PROJECT_PATH, rename file to `.env`
 2. To download and prepare data, use `make data_mnist` or `make data_oxford`
 3. To run the classification example, use `python run_training.py mode=exp name=exp_test`.
 4. To run the segmentation example, use `python run_training.py --config-name config_seg mode=exp name=exp_test`.


## Project Organization

<details>
<summary><b>Show project structure</b></summary>

```
├──.venv                        <- Local poetry environment
│   └──.gitkeep
├── configs                     <- Hydra configuration files
│   ├── callbacks               <- Callbacks configs
│   ├── datamodule              <- Datamodule configs
│   │   └── data_aug            <- TODO
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
│                           the creator's initials, and a short `-` delimited description, e.g.
│                           `1.0-jqp-initial-data-exploration`.
├── references          <- Data dictionaries, manuals, and all other explanatory materials.
├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures         <- Generated graphics and figures to be used in reporting
├── test                <- TODO
├── {{cookiecutter.module_name}}              <- Source code for use in this project.
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
│   │   └── training.py                   <- TODO
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
│   │   ├── my_losses.py                  <- TODO
│   │   ├── my_metrics.py                 <- TODO
│   │   └── utils.py                      <- TODO
│   │
│   ├── visualization                     <- Scripts to create exploratory and results oriented
│   │   └── __init__.py                    visualizations
│   │
│   └── __init__.py                       <- Makes {{cookiecutter.module_name}} a Python module
│
├── .env.example            <- TODO
├── .editorconfig           <- file with format specification. You need to install
│                               the required plugin for your IDE in order to enable it.
├── .gitignore              <- file that specifies what should we commit into
│                               the repository and we should not.
├── .pre-commit-config.yaml <- TODO
├── LICENSE
├── Makefile                <- Makefile with commands like `make data_mnist`
├── poetry.toml             <- poetry config file to install enviroment locally
├── poetry.lock             <- lock file for dependencies. It is used to install exactly
│                               the same versions of dependencies on each build
├── pyproject.toml          <- The project's dependencies for reproducing the
│                               analysis environment
├── README.md               <- The top-level README for developers using this project.
├── run_training.py         <- TODO
└── setup.cfg               <- configuration file, that is used by most tools in this project
```

</details>

## Guide

### How To Get Started

- First, you should probably get familiar with [Tensorflow](https://www.tensorflow.org/)
- Next, go through [Hydra quick start guide](https://hydra.cc/docs/intro/) and [basic Hydra tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/)
  <br>

### How it works

By design, every run is initialized by [run_training.py](run_training.py) file. All modules are dynamically instantiated from module paths specified in config. Example model config:

```yaml
_target_: {{cookiecutter.module_name}}.model.modules.simple_dense_net.SimpleDenseNet
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
The whole pipeline managing the instantiation logic is placed in [{{cookiecutter.module_name}}/executor/training.py]({{cookiecutter.module_name}}/executor/training.py).

<br>

### Main Project Configuration

Location: [configs/config.yaml](configs/config.yaml)<br>
Main project config contains default training configuration.<br>
It determines how config is composed when simply executing command `python run_training.py`.<br>
It also specifies everything that shouldn't be managed by experiment configurations.

<details>
<summary><b>Show main project configuration</b></summary>

```yaml
# specify here default training configuration
defaults:
  - trainer: default.yaml
  - model: mnist_model.yaml
  - datamodule: mnist_datamodule.yaml
  - callback: default.yaml # set this to null if you don't want to use callback
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

seed: "OxCAFFEE"

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
<br>

### Workflow

1. Write your model (see [simple_conv_net.py]({{cookiecutter.module_name}}/model/modules/simple_conv_net.py) for example)
2. Write your datamodule (see [mnist_datamodule.py]({{cookiecutter.module_name}}/datamodules/mnist_datamodule.py) for example)
3. Write your experiment config, containing paths to your model and datamodule
4. Run training with chosen experiment config: `python run_training.py mode=exp experiment=[your_config_name]`
   <br>

### Logs

Hydra creates new working directory for every executed run. <br>
By default, logs have the following structure:

```
│
├── logs
│   ├── experiments                         # Folder for logs generated from single runs
│   │   ├── exp_test                        # Name of your experiment
│   │   │   ├── 2021-02-15_16-50-49         # Date and Hour of executing run
│   │   │   │   ├── hydra_training          # Hydra logs
│   │   │   │   ├── checkpoints             # Training checkpoints
│   │   │   │   └── ...                     # Any other thing saved during training
│   │   │   ├── ...
│   │   │   └── ...
│   │   ├── ...
│   │   └── multiruns                             # Folder for logs generated from multiruns (sweeps)
│   │           ├── exp_test                      # Name of your experiment
│   │           │   ├── 2021-02-15_16-50-49       # Date and Hour of executing run
│   │           │   │   ├── 0                     # Job number
│   │           │   │   │   ├── hydra_training    # Hydra logs
│   │           │   │   │   ├── checkpoints       # Training checkpoints
│   │           │   │   │   └── ...               # Any other thing saved during training
│   │           │   │   ├── 1
│   │           │   │   ├── 2
│   │           │   │   └── ...
│   │           │   ├── ...
│   │           │   └── ...
│   │           ├── ...
│   │           └── ...
│   ├── debug                         # Folder for logs generated from debug runs

```


### Based on:
[cookiecutter-deep-learning-template](https://github.com/Ivo-B/CC-DL-template)

[cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science)

[wemake-django-template](https://github.com/wemake-services/wemake-django-template)

[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
