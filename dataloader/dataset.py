# Torch
from torch.utils.data import DataLoader

# Custom modules
from dataloader.puma_dataset import PumaDataset

def get_dataloader(
    image_dir=None,
    geojson_dir=None,
    run_mode='train',
    batch_size=1,
    transform=None
):
    """
    Returns a DataLoader object for the PUMA dataset.
    """

    dataset = PumaDataset(
        image_dir=image_dir,
        geojson_dir=geojson_dir,
        transform=transform
    )

    shuffle = True if run_mode == "train" else False
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader
