:: Download PUMA dataset
wget https://zenodo.org/records/14213079/files/01_training_dataset_geojson_nuclei.zip?download=1 -O data/01_training_dataset_geojson_nuclei.zip
wget https://zenodo.org/records/14213079/files/01_training_dataset_tif_ROIs.zip?download=1 -O data/01_training_dataset_tif_ROIs.zip

:: Unzip
tar -xf data/01_training_dataset_geojson_nuclei.zip -C data
tar -xf data/01_training_dataset_tif_ROIs.zip -C data

:: Remove zip files
rm data/01_training_dataset_geojson_nuclei.zip
rm data/01_training_dataset_tif_ROIs.zip

:: Download pre-trained resnet model
wget https://download.pytorch.org/models/resnet50-0676ba61.pth -O pretrained/resnet50-0676ba61.pth