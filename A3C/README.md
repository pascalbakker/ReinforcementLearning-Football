# ReinforcementLearning-Football

Reinforcement Learning Final Project


## Installing

Use Docker:

### Create image with GPU

docker build --build-arg DOCKER_BASE=tensorflow/tensorflow:1.15.2-gpu-py3 --build-arg DEVICE=gpu . -t gfootball

### Run:

docker run --gpus all -e DISPLAY=$DISPLAY -it -v /tmp/.X11-unix:/tmp/.X11-unix:rw gfootball bash

[GFootBall Source](https://github.com/google-research/football/blob/master/gfootball/doc/docker.md).

"If you get errors related to --gpus all flag, you can replace it with --device /dev/dri/[X] adding this flag for every file in the /dev/dri/ directory. It makes sure that GPU is visibile inside the Docker image. You can also drop it altogether (environment will try to perform software rendering)."

## BUILD OUR PROJECT

sudo docker build -t ri_final_project .

## TRAIN OUR PROJECT

sudo docker run -it ri_final_project main.py


