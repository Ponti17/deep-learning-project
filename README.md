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

```
> train_puma.py --configs configs/config.yml --neptune_project neptune-workspace/neptune-project
```

## Inference

Unfortunately we did not have time to implement a nice and shiny inference script. Look in our messy notebooks to see how this works.

## Compiling Report

The report can be compiled with

```bash
typst compile typst/main.typ
```