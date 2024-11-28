import argparse
import os

import torch
import torch.optim as optim

import neptune
from neptune_pytorch import NeptuneLogger
from neptune.utils import stringify_unsupported

from hover_net.dataloader.dataset import get_dataloader
from hover_net.models import HoVerNetExt
from hover_net.process import proc_valid_step_output, train_step, valid_step
from hover_net.tools.utils import (dump_yaml, read_yaml,
                                   update_accumulated_output)

def get_dir():
    """
    Returns the directory of main
    """
    return os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train model with PanNuck dataset")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="yaml config file path"
    )
    args = parser.parse_args()

    config = read_yaml(args.config)

    train_dataloader = get_dataloader(
        dataset_type="consep",
        data_path=config["DATA"]["TRAIN_DATA_PATH"],
        with_type=True,
        input_shape=(
            config["DATA"]["PATCH_SIZE"],
            config["DATA"]["PATCH_SIZE"]
        ),
        mask_shape=(
            config["DATA"]["PATCH_SIZE"],
            config["DATA"]["PATCH_SIZE"]
        ),
        batch_size=config["TRAIN"]["BATCH_SIZE"],
        run_mode="train",
    )
    val_dataloader = get_dataloader(
        dataset_type="consep",
        data_path=config["DATA"]["VALID_DATA_PATH"],
        with_type=True,
        input_shape=(
            config["DATA"]["PATCH_SIZE"],
            config["DATA"]["PATCH_SIZE"]
        ),
        mask_shape=(
            config["DATA"]["PATCH_SIZE"],
            config["DATA"]["PATCH_SIZE"]
        ),
        batch_size=config["TRAIN"]["BATCH_SIZE"],
        run_mode="val",
    )

    model = HoVerNetExt(
        backbone_name=config["MODEL"]["BACKBONE"],
        pretrained_backbone=config["MODEL"]["PRETRAINED"],
        num_types=config["MODEL"]["NUM_TYPES"],
    )
    optimizer = optim.Adam(model.parameters(), lr=1.0e-4, betas=(0.9, 0.999))

    model.to(config["TRAIN"]["DEVICE"])

    os.makedirs(config["LOGGING"]["SAVE_PATH"], exist_ok=True)
    dump_yaml(
        os.path.join(
            config["LOGGING"]["SAVE_PATH"],
            "config.yaml"
        ),
        config
    )

    neptune_api_token = open(f"{get_dir()}/neptune_api.key", "r", encoding="utf-8").read().strip()

    # Initialize neptune.ai
    run = neptune.init_run(
        project="ponti-workspace/hover-net",
        api_token=neptune_api_token
    )

    npt_logger = NeptuneLogger(
        run=run,
        model=model,
        log_parameters=True,
        log_freq=1
    )

    for epoch in range(config['TRAIN']['EPOCHS']):
        accumulated_output = {}
        for step_idx, data in enumerate(train_dataloader):
            train_result_dict = train_step(
                epoch,
                step_idx,
                batch_data=data,
                model=model,
                optimizer=optimizer,
                device=config["TRAIN"]["DEVICE"],
                show_step=1,
                verbose=config["LOGGING"]["VERBOSE"],
                npt_logger=npt_logger,
                run=run,
            )

        for step_idx, data in enumerate(val_dataloader):
            valid_result_dict = valid_step(
                epoch,
                step_idx,
                batch_data=data,
                model=model,
                device=config["TRAIN"]["DEVICE"]
            )
            update_accumulated_output(accumulated_output, valid_result_dict)

        out_dict = proc_valid_step_output(accumulated_output)

        print(
            f"[Epoch {epoch + 1} / {config['TRAIN']['EPOCHS']}] Val || "
            f"ACC={out_dict['scalar']['np_acc']:.3f} || "
            f"DICE={out_dict['scalar']['np_dice']:.3f} || "
            f"MSE={out_dict['scalar']['hv_mse']:.3f}"
        )

        if (epoch + 1) % config["LOGGING"]["SAVE_STEP"] == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    config["LOGGING"]["SAVE_PATH"],
                    f"epoch_{epoch + 1}.pth"
                )
            )

    torch.save(
        model.state_dict(),
        os.path.join(config["LOGGING"]["SAVE_PATH"], "latest.pth")
    )

    npt_logger.log_model("model")
    run.stop()
