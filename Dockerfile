FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
	python3-dev python3-pip git sudo
RUN python3 -m pip install --break-system-packages torch
RUN python3 -m pip install --break-system-packages tqdm jupyterlab
RUN python3 -m pip install --break-system-packages matplotlib
RUN python3 -m pip install --break-system-packages transformers
RUN python3 -m pip install --break-system-packages mlflow
RUN python3 -m pip install --break-system-packages pandas
RUN python3 -m pip install --break-system-packages scikit-learn

# create a non-root user
ARG USER_ID=999
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
