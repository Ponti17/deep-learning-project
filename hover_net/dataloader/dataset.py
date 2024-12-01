from torch.utils.data import DataLoader

from hover_net.datasets.consep_dataset import CoNSePDataset
from hover_net.datasets.inference_dataset import (FolderInferenceDataset,
                                                  SingleInferenceDataset)


def get_dataloader(
    dataset_type=None,
    data_path=None,
    with_type=True,
    input_shape=None,
    mask_shape=None,
    batch_size=1,
    run_mode="train",
):
    """
    Get dataloader for training, validation, inference.

    When run_mode is "train" or "val", the dataloader is created for training
    or validation. When run_mode is "inference_folder", the dataloader is
    created for inference on several image in a folder. When run_mode is 
    "inference_single", the dataloader is created for inference on a single image.
    """
    if run_mode == "inference_folder":
        dataset = FolderInferenceDataset(
            data_path=data_path, input_shape=input_shape
        )
    elif run_mode == "inference_single":
        dataset = SingleInferenceDataset(
            data_path_list=data_path, input_shape=input_shape
        )
    elif dataset_type.lower() == "consep":
        dataset = CoNSePDataset(
            data_path=data_path,
            with_type=with_type,
            input_shape=input_shape,
            mask_shape=mask_shape,
            run_mode=run_mode,
            setup_augmentor=True,
        )
    else:
        raise NotImplementedError

    shuffle = True if run_mode == "train" else False
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
    return dataloader
