#!/bin/bash

# Install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh 
chmod +x Anaconda3-2019.10-Linux-x86_64.sh
./Anaconda3-2019.10-Linux-x86_64.sh

source ~/.bashrc                        # Load in the new bashrc
conda env create --file environment.yml # Create the new env
conda activate birdear                  # Activate the env
