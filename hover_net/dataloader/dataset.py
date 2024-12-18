# Packages
from torch.utils.data import DataLoader

# HoVer Net
from hover_net.datasets.puma_dataset import PumaDataset

def get_dataloader(
    image_path   = None,
    geojson_path = None,
    input_shape  = None,
    mask_shape   = None,
    batch_size   = 1,
    run_mode     = "train",
):
    """
    Get dataloader for training, validation, inference.

    Args:
        - image_path (str): Path to image directory.
        - geojson_path (str): Path to geojson directory.
        - with_type (bool): Whether to include type information.
        - input_shape (tuple): Shape of input image.
        - mask_shape (tuple): Shape of mask.
        - batch_size (int): Batch size.
        - run_mode (str): Mode of operation.
    """
    # We have called this function with bad arguments so many times
    # Raise error if we do it again!
    if image_path is None:
        raise ValueError("Image path is required.")
    if geojson_path is None:
        raise ValueError("Geojson path is required.")
    if input_shape is None:
        raise ValueError("Input shape is required.")
    if mask_shape is None:
        raise ValueError("Mask shape is required.")
    if run_mode not in ["train", "valid", "test"]:
        raise ValueError("Invalid run_mode. Must be one of 'train', 'valid', 'test'.")

    dataset = PumaDataset(
        image_path=image_path,
        geojson_path=geojson_path,
        input_shape=input_shape,
        mask_shape=mask_shape,
        run_mode=run_mode,
        augment=True,
        )

    # Shuffle the dataset if we are in training mode
    shuffle = True if run_mode == "train" else False
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )

    return dataloader
