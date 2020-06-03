# BirdEar
Birdear is an open source hardware project that aims to build embedded hardware for classifying acoustic data. It uses and STM32 microcontoller, 18650 battery and a neural network build woth [Tensorflow lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers). It is designed to be inexpensive so several units can be placed cheaply.

The training data was fetched from [xeno-canto](https://www.xeno-canto.org/)

![](report/img/E04A0175.JPG)

[Report here](report/paper.pdf)

## Repo Structure
```bash
├── birdEar   # Hardware directory
├── NN        # Directory pertaining to the neural net
├── README.md # This file
├── refs      # Referance files
├── report    # Report for the project
└── src       # Source code for the firmware
```

- Jin Seo / Ryan Walker
