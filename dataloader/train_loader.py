import os
import json

# Misc
import numpy as np

# ML
import torch
from torchvision.transforms import ToTensor

# GeoJSON
from shapely.geometry import shape
from shapely.geometry.polygon import Polygon
from PIL import Image, ImageDraw

class PumaDataset(torch.utils.data.Dataset):
    """
    Dataloader for the PUMA dataset.
    """
    def __init__(self, image_dir, geojson_dir, transform=None):
        """        
        Args:
        - image_dir (string): Directory with all the images.
        - geojson_dir (string): Directory with all the geojson files.
        - transform_image (callable, optional): Optional transform to be applied on an image.
        - transform_geojson (callable, optional): Optional transform to be applied on a geojson.
        """
        self.image_dir      = image_dir
        self.geojson_dir    = geojson_dir
        self.images         = os.listdir(image_dir)
        self.geojsons       = os.listdir(geojson_dir) #? Is geojsons as word?
        self.transform      = transform

        # Polygon class labels
        self.classes = {
            'nuclei_tumor':        0, # Tumor
            'nuclei_lymphocyte':   1, # TIL
            'nuclei_plasma_cell':  1, # TIL
            'nuclei_endothelium':  2, # Other
            'nuclei_apoptosis':    2, # Other
            'nuclei_stroma':       2, # Other
            'nuclei_histiocyte':   2, # Other
            'nuclei_melanophage':  2, # Other
            'nuclei_neutrophil':   2, # Other
            'nuclei_epithelium':   2, # Other
        }

    def __len__(self):
        """
        Returns the number of images in the datset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrives an image and its corresponding mask.
        
        Args:
        - idx (int): Index of the image to be retrieved.
        
        Returns:
        - image: Transformed image tensor.
        - mask: Transformed mask tensor.
        """
        # Load image
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(img_path)
        width, height = img.size

        # Create 3 blank greyscale image to draw polygons on
        img_poly_0 = Image.new(mode="L", size=(width, height)) # Tumor
        img_poly_1 = Image.new(mode="L", size=(width, height)) # TIL
        img_poly_2 = Image.new(mode="L", size=(width, height)) # Other
        draw_0 = ImageDraw.Draw(img_poly_0) # Tumor
        draw_1 = ImageDraw.Draw(img_poly_1) # TIL
        draw_2 = ImageDraw.Draw(img_poly_2) # Other

        # Load geojson
        #! Fold hands and pray to the lord that the images and geojsons are in the same order
        json_path = os.path.join(self.geojson_dir, self.geojsons[idx])
        with open(json_path, encoding='utf-8') as f:
            geojson = json.load(f)

        legend = {}

        # Perform the GeoJSON besværgelse
        for feature in geojson['features']:
            geometry = shape(feature['geometry'])
            label = feature['properties']['classification']["name"]
            color = feature['properties']['classification']["color"]

            legend[label] = color

            #! We do not support MultiPolygon
            if geometry.geom_type == 'Polygon':
                coords = geometry.exterior.coords
                if label == 'nuclei_tumor':
                    draw_0.polygon(coords, outline=255, fill=255)
                elif label == 'nuclei_lymphocyte' or label == 'nuclei_plasma_cell':
                    draw_1.polygon(coords, outline=255, fill=255)
                else:
                    draw_2.polygon(coords, outline=255, fill=255)

        # Albumentations KEEPS the mask dimensions (H, W, C) but changes the image to (C, H, W).
        # This is because PyTorch expects images to be channel first.
        mask = np.stack([np.array(img_poly_0), np.array(img_poly_1), np.array(img_poly_2)], axis=0)
        mask = np.transpose(mask, (1, 2, 0))  # (H, W, 3)
        img = np.array(img)[:, :, :3]         # (H, W, 3)|

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        else:
            raise ValueError("Transformations must be provided")

        return img, mask
