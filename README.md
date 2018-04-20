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

Be aware of the fact that the script will overwrite conda enviroments called `tensorflow`

## References
* Talk given 2017.11, see https://gitlab.lhotse.ov.otto.de/tesla/information-hub/tree/master/presentations/17_11_bob-andrews.