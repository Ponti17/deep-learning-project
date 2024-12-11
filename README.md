# Deep Learning Project

PyTorch implementation of HoVer-Net.

For more information on the PUMA challenge, visit the [website](https://puma.grand-challenge.org/).

**Authors**:

- Andreas Pedersen

- Lukas Hedegaard

## Repository Structure

The overall repository structure is listed below.

```
deep-learning-project:
+---configs
+---data
+---hover_net
|   +---dataloader
|   +---datasets
|   +---models
|   +---postprocess
|   +---process
|   \---tools
+---notebooks
+---presentation
+---pretrained
\---typst
```

- `root/`: contains the `train_puma.py` script for training and the `raytune.py` script for optimizing.

- `configs/`: directory for config yml files specifying model and train parameters.

- `hover_net`: contains everything related to the HoVer-Net, training, validation etc.

- `notebooks`: a collection of messy Python notebooks used when creating the project.

- `presentation`: our pitch presentation.

- `pretrained`: empty directory for placement of pretrained ResNets.

- `typst`: contains our report in Typst format.

## Files

- `hover_net/dataloader/dataset.py`: contains a function `get_dataloader` that returns dataloader that returns HoVer-Net targets generated from the PUMA dataset.

- `hover_net/dataloader/preprocessing.py/`: contains the functions that generate HoVer-Net targets from an instance map from the dataloader.

- `hover_net/datasets/puma_dataset.py`: the dataloader for the PUMA dataset. Parses GeoJSONs. This also contains our augmentation pipeline.

- `hover_net/models/`: this entire directory contains the net description. Not much to say about this as we use pretty much the same net as in the original paper. These files were barely touched by us.

- `hover_net/postprocess/postproces.py`: contains code to process the HoVer-Net predictions. Generates predicted instance map from HV and NP maps. Also generates a dictionary of all nuclei instances.

- `hover_net/process/infer.py`: contains code to infer a batch of images.

- `hover_net/process/train.py`: contains the train step. Called for every step of the training loop.

- `hover_net/process/utils.py`: contains a function for processing data from the validation loop. Calculates dice coefficients.

- `hover_net/process/validate.py`: contains the valid step. Called for every step of the validation loop.

- `hover_net/tools/utils.py`: contains utility functions to load/dump `.yml` files and update a dict of accumulated output.

## Getting Started

To get started you first must acquire the dataset and the weights for a pre-trained ResNet50 (unless using ResNeXt). You can get everything setup by running:

**Windows**
```cmd
> setup.bat
```

**Linux**
```bash
> setup.sh
```

The setup script will download and extract the dataset, as well as download weights for a ResNet. This is particularly useful if you're working on a compute cluster (such as LUMI).

### Config

A template `config.yml` file is already present in `configs/`. Change it to suit your needs.

### Training

To start training, run:

```cmd
> train_puma.py --configs configs/config.yml --neptune_project neptune-workspace/neptune-project
```

### Neptune

To log with Neptune you must save your api key in a file called `neptune_api.key` in the root of the project.

## Inference

Unfortunately we did not have time to implement a nice and shiny inference script. Look in our messy notebooks to see how this works.

## Compiling Report

The report can be compiled with

```cmd
typst compile typst/main.typ
```