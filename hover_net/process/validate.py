from collections import OrderedDict

import torch
import torch.nn.functional as F

from hover_net.models.loss import dice_loss, mse_loss, msge_loss, xentropy_loss

def valid_step(
    batch_data,
    model,
    run,
    loss_opts,
    device="cuda",
):
    """
    Validate the hover-net with Neptune logging.
    """
    # Put model in evaluation
    model.eval()

    loss_func_dict = {
        "bce": xentropy_loss,
        "dice": dice_loss,
        "mse": mse_loss,
        "msge": msge_loss,
    }

    result_dict = {"EMA": {}}

    def track_value(name, value):
        result_dict["EMA"].update({name: value})
        run[f"training/{name}"].append(value)

    imgs = batch_data["img"]
    true_np = batch_data["np_map"]
    true_hv = batch_data["hv_map"]

    imgs = imgs.to(device).type(torch.float32).permute(0, 3, 1, 2).contiguous()

    true_np = true_np.to(device).type(torch.int64)
    true_hv = true_hv.to(device).type(torch.float32)

    true_np_onehot = F.one_hot(true_np, num_classes=2).type(torch.float32)
    true_dict = {
        "np": true_np_onehot, 
        "hv": true_hv,
        }

    if model.num_types is not None:
        true_tp = batch_data["tp_map"].to(device).type(torch.int64)
        true_tp_onehot = F.one_hot(true_tp, num_classes=model.num_types).type(torch.float32)
        true_dict["tp"] = true_tp_onehot

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        pred_dict = model(imgs)
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
        if model.num_types is not None:
            pred_dict["tp"] = F.softmax(pred_dict["tp"], dim=-1)

        loss = 0
        for branch_name in pred_dict.keys():
            for loss_name, loss_weight in loss_opts[branch_name].items():
                loss_func = loss_func_dict[loss_name]
                loss_args = [true_dict[branch_name], pred_dict[branch_name]]
                if loss_name == "msge":
                    loss_args.extend([true_np_onehot[..., 1], device])
                term_loss = loss_func(*loss_args)
                track_value(f"loss_{branch_name}_{loss_name}", term_loss.cpu().item())
                loss += loss_weight * term_loss

        track_value("overall_loss", loss.cpu().item())

    return result_dict
