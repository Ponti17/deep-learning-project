import os
import argparse

# ML
import torch
import torch.optim as optim

# Custom modules
from hover_net.dataloader.dataset import get_dataloader
from hover_net.models import HoVerNetExt
from hover_net.process import proc_valid_step_output, train_step, valid_step 
from hover_net.tools.utils import (dump_yaml, read_yaml, update_accumulated_output)

import neptune
from neptune.types import File


def get_dir():
    """
    Returns the directory of main
    """
    return os.path.dirname(os.path.realpath(__file__))



def main():
    """
    Main function to train model with PUMA dataset
    """
    # User must parse the config file
    parser = argparse.ArgumentParser("Train model with PUMA dataset")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="yaml config file path"
    )
    parser.add_argument(
        "--neptune_project",
        type=str,
        required=False,
        help="Neptune project name (optional, defaults to env NEPTUNE_PROJECT)"
    )
    args = parser.parse_args()

    config = read_yaml(args.config)


    # Load the Neptune API key from the file
    with open("neptune_api.key", "r") as f:
        neptune_api_key = f.read().strip()

    # Determine the Neptune project
    neptune_project = args.neptune_project or os.getenv("NEPTUNE_PROJECT")
    if not neptune_project:
        raise ValueError(
            "Neptune project name is required. "
            "Provide it via the --neptune_project argument or set the NEPTUNE_PROJECT environment variable."
        )

    # Initialize Neptune
    run = neptune.init_run(
        name=config['LOGGING']['RUN_NAME'],
        project=neptune_project,
        api_token=neptune_api_key
    )
    run["config"] = config

    loss_opts = {
        "np": {"bce": 0.56, "dice": 1.29},
        "hv": {"mse": 1.5, "msge": 1.9},
        "tp": {"bce": 1.46, "dice": 1.52},
    }

    # Training and Validation Loops
    train_dataloader = get_dataloader(
        image_path=config["DATA"]["IMAGE_PATH"],
        geojson_path=config["DATA"]["GEOJSON_PATH"],
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
        image_path=config["DATA"]["IMAGE_PATH"],
        geojson_path=config["DATA"]["GEOJSON_PATH"],
        input_shape=(
            config["DATA"]["PATCH_SIZE"],
            config["DATA"]["PATCH_SIZE"]
        ),
        mask_shape=(
            config["DATA"]["PATCH_SIZE"],
            config["DATA"]["PATCH_SIZE"]
        ),
        batch_size=config["TRAIN"]["BATCH_SIZE"],
        run_mode="test",
    )

    model = HoVerNetExt(
        backbone_name=config["MODEL"]["BACKBONE"],
        pretrained_backbone=config["MODEL"]["PRETRAINED"],
        num_types=config["MODEL"]["NUM_TYPES"],
        freeze=False
    )

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.00002)

    model.to(config["TRAIN"]["DEVICE"])

    os.makedirs(config["LOGGING"]["SAVE_PATH"], exist_ok=True)
    dump_yaml(
        os.path.join(
            config["LOGGING"]["SAVE_PATH"],
            "config.yaml"
        ),
        config
    )

    for epoch in range(config['TRAIN']['EPOCHS']):
        run["training/epoch"].log(epoch + 1)
        # Training loop
        accumulated_output = {}
        for step_idx, data in enumerate(train_dataloader):
            train_result_dict = train_step(
                epoch,
                step_idx,
                batch_data=data,
                model=model,
                optimizer=optimizer,
                loss_opts=loss_opts,
                device=config["TRAIN"]["DEVICE"],
                run=run  # Pass Neptune run
            )

        # Validation loop
        for step_idx, data in enumerate(val_dataloader):
            valid_result_dict = valid_step(
                batch_data=data,
                model=model,
                device=config["TRAIN"]["DEVICE"]
            )
            update_accumulated_output(accumulated_output, valid_result_dict)

        lr_scheduler.step()
        out_dict = proc_valid_step_output(accumulated_output, nr_types=config["MODEL"]["NUM_TYPES"])

        # Log validation metrics to Neptune
        run["validation/accuracy"].log(out_dict["scalar"]["np_acc"])
        run["validation/dice"].log(out_dict["scalar"]["np_dice"])
        run["validation/mse"].log(out_dict["scalar"]["hv_mse"])
        run["validation/tp_dice_1"].log(out_dict["scalar"]["tp_dice_1"])
        run["validation/tp_dice_2"].log(out_dict["scalar"]["tp_dice_2"])
        run["validation/tp_dice_3"].log(out_dict["scalar"]["tp_dice_3"])

        print(
            f"[Epoch {epoch + 1} / {config['TRAIN']['EPOCHS']}] Val || "
        )

        # Save model periodically and log to Neptune
        if (epoch + 1) % config["LOGGING"]["SAVE_STEP"] == 0:
            model_path = os.path.join(
                config["LOGGING"]["SAVE_PATH"],
                f"epoch_{epoch + 1}.pth"
            )
            torch.save(model.state_dict(), model_path)

    # Save the final model
    final_model_path = os.path.join(config["LOGGING"]["SAVE_PATH"], "latest.pth")
    torch.save(model.state_dict(), final_model_path)
    run["model/latest"].upload(File(final_model_path))

    # Stop Neptune run
    run.stop()


if __name__ == "__main__":
    main()