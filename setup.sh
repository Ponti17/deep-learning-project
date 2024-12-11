#!/bin/bash

# Download PUMA dataset
echo "Downloading PUMA dataset..."
wget https://zenodo.org/records/14213079/files/01_training_dataset_geojson_nuclei.zip?download=1 -O data/01_training_dataset_geojson_nuclei.zip
wget https://zenodo.org/records/14213079/files/01_training_dataset_tif_ROIs.zip?download=1 -O data/01_training_dataset_tif_ROIs.zip

# Unzip
echo "Unzipping datasets..."
unzip -o data/01_training_dataset_geojson_nuclei.zip -d data
unzip -o data/01_training_dataset_tif_ROIs.zip -d data

# Remove zip files
echo "Cleaning up zip files..."
rm data/01_training_dataset_geojson_nuclei.zip
rm data/01_training_dataset_tif_ROIs.zip

# Download pre-trained ResNet model
echo "Downloading pre-trained ResNet model..."
wget https://download.pytorch.org/models/resnet50-0676ba61.pth -O pretrained/resnet50-0676ba61.pth

echo "Download and setup complete."
