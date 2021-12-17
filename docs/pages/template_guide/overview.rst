Overview
========


System requirements
-------------------

- ``git`` with a version at least ``2.16`` or higher
- ``docker`` with a version at least ``18.02`` or higher
- ``python`` with exact version, see ``pyproject.toml``
- ``poetry`` with a version at least ``1.1.2`` or higher


Architecture
------------

config
~~~~~~

- ``.env.example`` - a basic example of what keys must be contained in
  your ``.env`` file, this file is committed to VCS
  and must not contain private or secret values
- ``.env`` - main file for secret configuration,
  contains private and secret values, should not be committed to VCS

root project
~~~~~~~~~~~~

- ``README.md`` - main readme file, it specifies the entry
  point to the project's documentation
- ``.editorconfig`` - file with format specification.
  You need to install the required plugin for your IDE in order to enable it
- ``.gitignore`` - file that specifies
  what should we commit into the repository and we should not
- ``run_training.py`` - main file for your ``training`` of the model.
- ``pyproject.toml`` - main file of the project.
  It defines the project's dependencies.
- ``poetry.lock`` - lock file for dependencies.
  It is used to install exactly the same versions of dependencies on each build
- ``setup.cfg`` - configuration file, that is used by all tools in this project

configs
~~~~~~~

- ``configs/config.yaml`` - TODO
- ``configs/callback/`` - TODO
- ``configs/datamodule/`` - TODO
- ``configs/datamodule/data_aug/`` - TODO
- ``configs/experiment/`` - TODO
- ``configs/hparams_search/`` - TODO
- ``configs/logger/`` - TODO
- ``configs/mode/`` - TODO
- ``configs/model/`` - TODO
- ``configs/trainer/`` - TODO
- ``configs/trainer/loss/`` - TODO
- ``configs/trainer/lr_scheduler/`` - TODO
- ``configs/trainer/metric/`` - TODO
- ``configs/trainer/optimizer/`` - TODO

cctest
~~~~~~

- ``cctest/__init__.py`` - package definition, empty file
- ``cctest/data/`` - TODO
- ``cctest/evaluation/`` - TODO
- ``cctest/executor/`` - TODO
- ``cctest/model/`` - TODO
- ``cctest/model/modules`` - TODO
- ``cctest/utils/`` - TODO
- ``cctest/visualization/`` - TODO

docker
~~~~~~

- ``Dockerfile`` - container definition, used both for development and production

tests
~~~~~

- ``tests/test_executor`` - tests that ensures that basic ``training``, ``testing``, and ``inference``
  stuff is working, should not be removed
- ``tests/test_datamodule`` - example tests for the ``Oxford pet`` dataloader, could be removed
- ``tests/conftest.py`` - main configuration file for ``pytest`` runner

docs
~~~~

- ``docs/Makefile`` - command file that builds the documentation for Unix
- ``docs/make.bat`` - command file for Windows
- ``docs/conf.py`` - ``sphinx`` configuration file
- ``docs/index.rst`` - main documentation file, used as an entry point
- ``docs/pages/project`` - folder that will contain
  documentation written by you!
- ``docs/pages/template`` - folder that contains documentation that
  is common for each project built with this template
- ``docs/documents`` - folder that should contain any documents you have:
  spreadsheets, images, requirements, presentations, etc
- ``docs/requirements.txt`` - helper file, contains dependencies
  for ``readthedocs`` service. Can be removed
- ``docs/README.rst`` - helper file for this directory,
  just tells what to do next
