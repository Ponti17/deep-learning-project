import os

# GeoJSON
import json
from shapely.geometry import shape
from shapely.geometry.polygon import Polygon
from PIL import Image, ImageDraw

import numpy as np
import albumentations as A

from hover_net.dataloader.preprocessing import cropping_center, gen_targets
from .hover_dataset import HoVerDatasetBase


class PumaDataset(HoVerDatasetBase):
    """
    Data loader for the PUMA dataset.

    Target images are generated from the GeoJSON files using ImageDraw.
    Images and targets are augmented using the albumentation library.
    After augmentation, horizontal and vertical maps are generated.

    The images and corresponding GeoJSON files are found by os.listdir().
    It is there VERY important that their naming schemes are the same so they
    are in the correct order.

    Args:
        image_path: path to the image directory
        geojson_path: path to the geojson directory
        input_shape: shape of the input [h,w] - defined in config.py
        mask_shape: shape of the output [h,w] - defined in config.py
        mode: 'train', 'valid'
    """

    def __init__(
        self,
        image_path,
        geojson_path,
        with_type=True,
        input_shape=None,
        mask_shape=None,
        run_mode="train",
        augment=True,
    ):
        if run_mode not in ["train", "valid", "test"]:
            raise ValueError("Invalid mode. Must be 'train', 'valid' or 'test'.")
        if input_shape is None or mask_shape is None:
            raise ValueError("input_shape and mask_shape must be defined.")
        
        self.augment = augment

        self.run_mode    = run_mode
        self.image_dir   = image_path
        self.geojson_dir = geojson_path

        self.images     = os.listdir(self.image_dir)
        self.geojsons   = os.listdir(self.geojson_dir)

        self.primary_rois_images      = sorted([image for image in self.images if 'primary' in image])
        self.metastatic_rois_images   = sorted([image for image in self.images if 'metastatic' in image])
        self.primary_rois_geojsons    = sorted([geojson for geojson in self.geojsons if 'primary' in geojson])
        self.metastatic_rois_geojsons = sorted([geojson for geojson in self.geojsons if 'metastatic' in geojson])

        num_pri = len(self.primary_rois_images)     # Number of primary ROIs
        num_met = len(self.metastatic_rois_images)  # Number of metastatic ROIsz
        train_idx_pri = int(0.7 * num_pri)          # Splitting index for train primary ROIs
        train_idx_met = int(0.7 * num_met)          # Splitting index for train metastatic ROIs
        valid_idx_pri = int(0.85 * num_pri)          # Splitting index for validation primary ROIs
        valid_idx_met = int(0.85 * num_met)          # Splitting index for validation metastatic ROIs

        print(f"Number of primary ROIs: {num_pri}")
        print(f"Number of metastatic ROIs: {num_met}")

        # 70% train, 15% val, 15% test
        if run_mode == "train":
            self.images = self.primary_rois_images[:train_idx_pri] + self.metastatic_rois_images[:train_idx_met]
            self.geojsons = self.primary_rois_geojsons[:train_idx_pri] + self.metastatic_rois_geojsons[:train_idx_met]
        elif run_mode == "valid":
            self.images = self.primary_rois_images[train_idx_pri:valid_idx_pri] + self.metastatic_rois_images[train_idx_met:valid_idx_met]
            self.geojson = self.primary_rois_geojsons[train_idx_pri:valid_idx_pri] + self.metastatic_rois_geojsons[train_idx_met:valid_idx_met]
        elif run_mode == "test":
            self.images = self.primary_rois_images[valid_idx_pri:] + self.metastatic_rois_images[valid_idx_met:]
            self.geojsons = self.primary_rois_geojsons[valid_idx_pri:] + self.metastatic_rois_geojsons[valid_idx_met:]

        # Polygon class labels
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

        if augment:
            self.setup_augmentor(run_mode)
        return

    def load_data(self, idx):
        """
        Loads an image an GeoJSON from specified path.

        A mask is created from the GeoJSON file, where each class is assigned
        a unique integer value. In addition a mask is created where the class
        of each nuclei is specified.
        """
        # Load image
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(img_path)
        width, height = img.size

        # Create blank images to draw instances and annotations
        instances   = Image.new(mode="L", size=(width, height))
        annotations = Image.new(mode="L", size=(width, height))
        inst_draw = ImageDraw.Draw(instances)
        ann_draw = ImageDraw.Draw(annotations)

        # Load GeoJSON
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

        # Combine instances and annotations
        ann = np.stack([np.array(instances), np.array(annotations)], axis=0).astype("int32")
        ann = np.transpose(ann, (1, 2, 0))
        img = np.array(img).astype("uint8")[:, :, :3]

        return img, ann

    def setup_augmentor(self, mode):
        self.augmentor = self.__get_augmentation(mode)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, ann = self.load_data(idx)

        if self.augment:
            augmented = self.augmentor(image=img, mask=ann)
            img = augmented["image"]
            ann = augmented["mask"]

        img = cropping_center(img, self.input_shape)
        feed_dict = {"img": img}

        inst_map = ann[..., 0]  # HW1 -> HW
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