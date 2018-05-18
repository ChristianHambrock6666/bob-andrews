# Pattern extraction from text

Convolutional network (CNN) trying to mimic a regular expression.
This is a proof of principle that a CNN is able to build text patterns by itself to extract information from a text to 
minimize a loss function.

## Prerequisites

An installation of the conda package manager is required.
A sufficient (according to needed packages in the LATEXWriter) installation of latex (lualatex) is needed.
The fastest way to obtain conda is to install [Miniconda](https://conda.io/miniconda.html).

## Setup 
A script for easy installation of the conda enviroments is included.
```bash
./setup.sh
```


## tensorboard
Beside LaTeX, we use tensorboard for visualization during training.  
You can activate it and make it visualize all stored log-files via, e.g., (adjust paths and envs):
```bash
bash -c "source /home/chambroc/miniconda3/bin/activate bob && tensorboard --logdir=./output/tensorboard"
```


## References
* see slides in doc folder