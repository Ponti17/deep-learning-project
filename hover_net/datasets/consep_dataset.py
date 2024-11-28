import os

# GeoJSON
import json
from shapely.geometry import shape
from shapely.geometry.polygon import Polygon
from PIL import Image, ImageDraw

import numpy as np
from imgaug import augmenters as iaa

from hover_net.dataloader.augmentation import (add_to_brightness,
                                               add_to_contrast, add_to_hue,
                                               add_to_saturation,
                                               gaussian_blur, median_blur)
from hover_net.dataloader.preprocessing import cropping_center, gen_targets

from .hover_dataset import HoVerDatasetBase


class CoNSePDataset(HoVerDatasetBase):
    """Data Loader. Loads images from a file list and
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are
    generated.
    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        mask_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'

    """

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
        # self.image_dir = image_dir
        # self.geojson_dir = geojson_dir

        self.image_dir = 'data/01_training_dataset_tif_ROIs'
        self.geojson_dir = 'data/01_training_dataset_geojson_nuclei'

        self.images     = os.listdir(self.image_dir)
        self.geojsons   = os.listdir(self.geojson_dir)

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
        self.id = 0
        if setup_augmentor:
            self.setup_augmentor(0, 0)
        return

    def setup_augmentor(self, worker_id, seed):
        self.augmentor = self.__get_augmentation(self.run_mode, seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        self.id = self.id + worker_id
        return

    def load_data(self, idx):
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

        print(ann.shape)
        print(img.shape)

        return img, ann

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, ann = self.load_data(idx)

        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            img = shape_augs.augment_image(img)
            ann = shape_augs.augment_image(ann)

        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            img = input_augs.augment_image(img)

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

    def __get_augmentation(self, mode, rng):
        if mode == "train":
            shape_augs = [
                # * order = ``0`` -> ``cv2.INTER_NEAREST``
                # * order = ``1`` -> ``cv2.INTER_LINEAR``
                # * order = ``2`` -> ``cv2.INTER_CUBIC``
                # * order = ``3`` -> ``cv2.INTER_CUBIC``
                # * order = ``4`` -> ``cv2.INTER_CUBIC``
                # ! for pannuke v0, no rotation or translation,
                # ! just flip to avoid mirror padding
                # iaa.Affine(
                #     # scale images to 80-120% of their size,
                #     # individually per axis
                #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                #     # translate by -A to +A percent (per axis)
                #     translate_percent={
                #         "x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                #     shear=(-5, 5),  # shear by -5 to +5 degrees
                #     rotate=(-179, 179),  # rotate by -179 to +179 degrees
                #     order=0,  # use nearest neighbour
                #     backend="cv2",  # opencv for fast processing
                #     seed=rng,
                # ),
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                ),
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
            ]

            input_augs = [
                iaa.OneOf(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: gaussian_blur(
                                *args, max_ksize=3
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: median_blur(
                                *args, max_ksize=3
                            ),
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                    ]
                ),
                iaa.Sequential(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_hue(
                                *args, range=(-8, 8)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_saturation(
                                *args, range=(-0.2, 0.2)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_brightness(
                                *args, range=(-26, 26)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_contrast(
                                *args, range=(0.75, 1.25)
                            ),
                        ),
                    ],
                    random_order=True,
                ),
            ]
        else:
            shape_augs = [
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                )
            ]
            input_augs = []

        return shape_augs, input_augs
