FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential python3.9 python3.9-venv python3.9-distutils vim git-all curl && \
  rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.9 /usr/bin/python

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.9 get-pip.py
# Poetry for Ubuntu
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -

ENV PATH="/root/.local/bin:$PATH"

ENV PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
    \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

RUN mkdir /docker_pwd

COPY poetry.toml pyproject.toml poetry.lock /docker_pwd/

WORKDIR /docker_pwd

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev
