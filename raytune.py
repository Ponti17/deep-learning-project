import os
import sys
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

# Raytune
import ray
from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch


def get_dir():
    """
    Returns the directory of main
    """
    return os.path.dirname(os.path.realpath(__file__))



def main(config, yml_config):
    """
    Main function to train model with PUMA dataset
    """
    run = None

    lr          = 0.000275
    weigth_decay = 3.1e-5
    np_bce     = config["np_bce"]
    np_dice    = config["np_dice"]
    hv_mse     = config["hv_mse"]
    hv_msge    = config["hv_msge"]
    tp_bce     = config["tp_bce"]
    tp_dice    = config["tp_dice"]
    step_size  = 16
    gamma      = 4.7e-5
    freeze     = False

    loss_opts = {
        "np": {"bce": np_bce, "dice": np_dice},
        "hv": {"mse": hv_mse, "msge": hv_msge},
        "tp": {"bce": tp_bce, "dice": tp_dice},
    }

    image_dir = os.path.join(get_dir(), yml_config["DATA"]["IMAGE_PATH"])
    geojson_dir = os.path.join(get_dir(), yml_config["DATA"]["GEOJSON_PATH"])

    # Training and Validation Loops
    train_dataloader = get_dataloader
        image_path=image_dir,
        geojson_path=geojson_dir,
        input_shape=(
            yml_config["DATA"]["PATCH_SIZE"],
            yml_config["DATA"]["PATCH_SIZE"]
        ),
        mask_shape=(
            yml_config["DATA"]["PATCH_SIZE"],
            yml_config["DATA"]["PATCH_SIZE"]
        ),
        batch_size=yml_config["TRAIN"]["BATCH_SIZE"],
        run_mode="train",
    )
    val_dataloader = get_dataloader(
        image_path=image_dir,
        geojson_path=geojson_dir,
        input_shape=(
            yml_config["DATA"]["PATCH_SIZE"],
            yml_config["DATA"]["PATCH_SIZE"]
        ),
        mask_shape=(
            yml_config["DATA"]["PATCH_SIZE"],
            yml_config["DATA"]["PATCH_SIZE"]
        ),
        batch_size=yml_config["TRAIN"]["BATCH_SIZE"],
        run_mode="test",
    )

    model = HoVerNetExt(
        backbone_name=yml_config["MODEL"]["BACKBONE"],
        pretrained_backbone=yml_config["MODEL"]["PRETRAINED"],
        num_types=yml_config["MODEL"]["NUM_TYPES"],
        freeze=freeze
    )

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weigth_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    model.to(yml_config["TRAIN"]["DEVICE"])

    for epoch in range(yml_config['TRAIN']['EPOCHS']):
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
                device=yml_config["TRAIN"]["DEVICE"],
                run=run  # Pass Neptune run
            )

        # Validation loop
        for step_idx, data in enumerate(val_dataloader):
            valid_result_dict = valid_step(
                batch_data=data,
                model=model,
                device=yml_config["TRAIN"]["DEVICE"]
            )
            update_accumulated_output(accumulated_output, valid_result_dict)

        lr_scheduler.step()
        out_dict = proc_valid_step_output(accumulated_output, nr_types=yml_config["MODEL"]["NUM_TYPES"])

        tp_dice = out_dict["scalar"]["tp_dice_1"] + out_dict["scalar"]["tp_dice_2"] + out_dict["scalar"]["tp_dice_3"]

    session.report({"tp_dice": tp_dice})

if __name__ == "__main__":
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

    this_path = get_dir()
    sys.path.append(this_path)

    yml_config = read_yaml(args.config)

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
        name=yml_config['LOGGING']['RUN_NAME'],
        project=neptune_project,
        api_token=neptune_api_key
    )
    run["config"] = yml_config

    #number of cpus:
    numGPUs=int(os.getenv('SLURM_GPUS_PER_TASK'))
    numCPUs=int(os.getenv('SLURM_CPUS_PER_TASK'))
    numDevices = {
        "numGPUs": numGPUs,
        "numCPUs": numCPUs
    }
    run["numDevices"] = numDevices

    search_space = {
        "np_bce":   tune.uniform(0.5, 2),
        "np_dice":  tune.uniform(0.5, 2),
        "hv_mse":   tune.uniform(0.5, 2),
        "hv_msge":  tune.uniform(0.5, 2),
        "tp_bce":   tune.uniform(0.5, 2),
        "tp_dice":  tune.uniform(0.5, 2),
    }

    algo = OptunaSearch()

    trainable_with_resources = tune.with_resources(tune.with_parameters(main, yml_config=yml_config),
        {
            "cpu": 1,
            "gpu": 1,
        }
    )

    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            metric="tp_dice",
            mode="max",
            search_alg=algo,
            num_samples=30,
            max_concurrent_trials=1
        ),
        run_config=air.RunConfig(
            stop={"training_iteration": 1},
        ),
        param_space=search_space,
    )

    # If I don't limit num_cpus, ray tries to use the whole node and crashes:
    ray.init(num_cpus=numCPUs, num_gpus=numGPUs, log_to_driver=False)

    result_grid = tuner.fit()
    print("Best config is:", result_grid.get_best_result().config,
    ' with accuracy: ', result_grid.get_best_result().metrics['tp_dice'])

    df = result_grid.get_dataframe()
    df.to_csv(os.path.join(yml_config['LOGGING']['SAVE_PATH'], "results.csv"), index=False)

    # Stop Neptune run
    run.stop()
