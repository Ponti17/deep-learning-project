import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torchviz
import os

# Importing neptune.ai
import neptune
from neptune_pytorch import NeptuneLogger
from neptune.utils import stringify_unsupported

# Importing the model
from net import Net

def get_dir():
    """
    Returns the directory of main
    """
    return os.path.dirname(os.path.realpath(__file__))

def main():
    """
    Main function for training the model
    """
    neptune_api_token = open(f"{get_dir()}/neptune_api.key", "r", encoding="utf-8").read().strip()

    # Initialize neptune.ai
    run = neptune.init_run(
        project="ponti-workspace/mnist-test",
        source_files=["python/main.py", "python/net.py"],
        api_token=neptune_api_token
    )

    params = {
        "lr": 1e-2,
        "batch_size": 64,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 20,
        "step_size": 1,
        "gamma": 0.7
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run["parameters"] = params

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST(f'{get_dir()}/../data', train=True, download=True,
                                transform=transform)
    dataset2 = datasets.MNIST(f'{get_dir()}/../data', train=False,
                                transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=params["batch_size"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=params["batch_size"], shuffle=True)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=params["lr"])

    npt_logger = NeptuneLogger(
        run=run,
        model=model,
        log_parameters=True,
        log_freq=30
    )

    run[npt_logger.base_namespace]["hyperparams"] = stringify_unsupported(params)

    scheduler = StepLR(optimizer, step_size=params["step_size"], gamma=params["gamma"])
    for epoch in range(1, params["epochs"] + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)

            # Log after every 30 steps
            if batch_idx % 30 == 0:
                run[npt_logger.base_namespace]["batch/loss"].append(loss.item())

            loss.backward()
            optimizer.step()
        run[npt_logger.base_namespace]["epoch/loss"].append(loss.item())
        npt_logger.log_checkpoint()
        scheduler.step()

    # End neptune.ai loggin
    npt_logger.log_model("model")
    run.stop()

if __name__ == "__main__":
    main()
