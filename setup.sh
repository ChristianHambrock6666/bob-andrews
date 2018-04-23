#!/bin/bash
conda create --name bob
source activate bob
conda env update --file ./environment.yml
conda install -n bob -c conda-forge ds-lime