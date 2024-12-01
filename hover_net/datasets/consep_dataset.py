import os

# GeoJSON
import json
from shapely.geometry import shape
from shapely.geometry.polygon import Polygon
from PIL import Image, ImageDraw

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from hover_net.dataloader.preprocessing import cropping_center, gen_targets

from .hover_dataset import HoVerDatasetBase


class CoNSePDataset(HoVerDatasetBase):
    """Data Loader using Albumentations for augmentations."""

    def __init__(
        self,
        data_path,
        with_type=False,
        input_shape=None,
        mask_shape=None,
        run_mode="train",
        setup_augmentor=True,
    ):
        assert input_shape is not None and mask_shape is not None
        self.run_mode = run_mode

        self.image_dir = 'data/01_training_dataset_tif_ROIs'
        self.geojson_dir = 'data/01_training_dataset_geojson_nuclei'

        self.images = os.listdir(self.image_dir)
        self.geojsons = os.listdir(self.geojson_dir)

        self.classes = {
            'nuclei_tumor': 0,  # Tumor
            'nuclei_lymphocyte': 1,  # TIL
            'nuclei_plasma_cell': 1,  # TIL
            'nuclei_endothelium': 2,  # Other
            'nuclei_apoptosis': 2,  # Other
            'nuclei_stroma': 2,  # Other
            'nuclei_histiocyte': 2,  # Other
            'nuclei_melanophage': 2,  # Other
            'nuclei_neutrophil': 2,  # Other
            'nuclei_epithelium': 2,  # Other
        }

        self.with_type = with_type
        self.mask_shape = mask_shape
        self.input_shape = input_shape
        if setup_augmentor:
            self.setup_augmentor(run_mode)

    def setup_augmentor(self, mode):
        self.augmentor = self.__get_augmentation(mode)

    def load_data(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(img_path)
        width, height = img.size

        instances = Image.new(mode="L", size=(width, height))
        annotations = Image.new(mode="L", size=(width, height))
        inst_draw = ImageDraw.Draw(instances)
        ann_draw = ImageDraw.Draw(annotations)

        json_path = os.path.join(self.geojson_dir, self.geojsons[idx])
        with open(json_path, encoding="utf-8") as f:
            geojson = json.load(f)

        inst_id = 1
        for feature in geojson["features"]:
            geometry = shape(feature["geometry"])
            label = feature["properties"]["classification"]["name"]

            if geometry.geom_type == "Polygon":
                coords = geometry.exterior.coords
                if label == "nuclei_tumor":
                    ann_draw.polygon(coords, outline=1, fill=1)
                elif label in ["nuclei_lymphocyte", "nuclei_plasma_cell"]:
                    ann_draw.polygon(coords, outline=2, fill=2)
                else:
                    ann_draw.polygon(coords, outline=3, fill=3)
                inst_draw.polygon(coords, outline=inst_id, fill=inst_id)
                inst_id += 1

        ann = np.stack([np.array(instances), np.array(annotations)], axis=0).astype("int32")
        ann = np.transpose(ann, (1, 2, 0))
        img = np.array(img).astype("uint8")[:, :, :3]

        return img, ann

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, ann = self.load_data(idx)

        if self.augmentor is not None:
            augmented = self.augmentor(image=img, mask=ann)
            img = augmented["image"]
            ann = augmented["mask"]

        img = cropping_center(img, self.input_shape)
        feed_dict = {"img": img}

        inst_map = ann[..., 0]
        if self.with_type:
            type_map = (ann[..., 1]).copy()
            type_map = cropping_center(type_map, self.mask_shape)
            feed_dict["tp_map"] = type_map

        target_dict = gen_targets(inst_map, self.mask_shape)
        feed_dict.update(target_dict)

        return feed_dict

    def __get_augmentation(self, mode):
        if mode == "train":
            aug = A.Compose(
                [
                    A.RandomCrop(height=self.input_shape[0], width=self.input_shape[1]),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=(3, 3), p=0.5),
                            A.MedianBlur(blur_limit=3, p=0.5),
                            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                        ],
                        p=0.5,
                    ),
                    A.OneOf(
                        [
                            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=8, val_shift_limit=8, p=0.5),
                            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                        ],
                        p=0.5,
                    ),
                ]
            )
        else:
            aug = A.Compose(
                [
                    A.CenterCrop(height=self.input_shape[0], width=self.input_shape[1]),
                ]
            )
        return aug
