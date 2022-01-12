FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04 

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential python3.9 python3.9-distutils vim git-all curl && \
  rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.9 get-pip.py

RUN ln -s /usr/bin/python3.9 /usr/bin/python

RUN mkdir /docker_pwd

COPY poetry.toml pyproject.toml poetry.lock /docker_pwd/

WORKDIR /docker_pwd

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

# RUN poetry --version
#
#
# Working Directory

