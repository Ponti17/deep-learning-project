from torch.utils.data import DataLoader

from hover_net.datasets.puma_dataset import PumaDataset

def get_dataloader(
    image_path   = None,
    geojson_path = None,
    with_type    = True,
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
        with_type=with_type,
        input_shape=input_shape,
        mask_shape=mask_shape,
        run_mode=run_mode,
        augment=True,
        )

    shuffle = True if run_mode == "train" else False
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
    return dataloader
