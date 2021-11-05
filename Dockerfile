FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04 

# RUN apt-get update && apt-get install -y --no-install-recommends \
#       ${NV_CUDNN_PACKAGE} \
#       && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} && \
#       rm -rf /var/lib/apt/lists/*
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential python3.9 python3.9-distutils vim git-all curl && \
  rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.9 get-pip.py

RUN ln -s /usr/bin/python3.9 /usr/bin/python
# RUN ln -s /usr/bin/pip3.9 /usr/bin/pip
# RUN echo $HOME

# RUN curl https://pyenv.run | bash

# ENV PATH="$HOME/.pyenv/bin:$PATH"

# ENV HOME="/root"
# WORKDIR ${HOME}
# ENV PYENV_ROOT="${HOME}/.pyenv"
# ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

# RUN eval "$(pyenv virtualenv-init -)" 

# ENV PYTHON_VERSION=3.9.7
# RUN pyenv install ${PYTHON_VERSION}
# RUN pyenv global ${PYTHON_VERSION}
#
# RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

# ENV PATH="/root/.poetry/bin:$PATH"


RUN pip install poetry
# RUN poetry --version

