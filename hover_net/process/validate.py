from collections import OrderedDict

import torch
import torch.nn.functional as F


def valid_step(epoch, model, validation_loader, loss_func_dict, device, run):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(validation_loader):
            imgs = batch_data["img"].to(device).type(torch.float32)
            imgs = imgs.permute(0, 3, 1, 2).contiguous()
            true_np = batch_data["np_map"].to(device).type(torch.int64)
            true_np_onehot = F.one_hot(true_np, num_classes=2).type(torch.float32)

            pred_dict = model(imgs)
            pred_dict = OrderedDict(
                [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
            )
            pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)

            batch_loss = 0
            for branch_name, pred in pred_dict.items():
                true = true_np_onehot if branch_name == "np" else batch_data["hv_map"].to(device).type(torch.float32)
                for loss_name, loss_func in loss_func_dict.items():
                    batch_loss += loss_func(true, pred)
            
            val_loss += batch_loss.item()

    avg_val_loss = val_loss / len(validation_loader)
    run["validation/avg_loss"].append(avg_val_loss)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss