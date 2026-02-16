#!/bin/bash

sudo apt-get update

# install docker
sudo apt install docker.io

# start docker service
sudo systemctl start docker
sudo systemctl enable docker

# install docker compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.25.5/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# build and run docker containers
cd ~/server
sudo docker-compose up --build
