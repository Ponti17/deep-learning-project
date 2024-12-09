# Core
from collections import OrderedDict

# Packages
import torch
import torch.nn.functional as F

def infer_step(batch_imgs, model, device="cuda"):
    """
    Infer a batch of images using the HoVer-net model.

    Args:
        - batch_imgs: torch.Tensor(B, H, W, C)
        - model: torch.nn.Module
    
    Returns:
        - pred_output: np.array(B, H, W, C)
            C=0: type map,
            C=1: nuclear pixel map,
            C=2: horizontal map,
            C=3: vertical map
    
    Based on: https://github.com/Kaminyou/HoVer-Net-PyTorch
    """
    # Move images to gpu and permute to (B, C, H, W)
    patch_imgs_gpu = batch_imgs.to(device).type(torch.float32)
    patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

    # Put model in eval mode
    model.eval()

    # ... And DONT compute gradients
    with torch.no_grad():
        # Forward pass
        # pred_dict: ordered dict with tp, np, hv keys
        pred_dict = model(patch_imgs_gpu)

        # Post-process the output
        # Permute back to (B, H, W, C)
        pred_list = []
        for k, v in pred_dict.items():
            pred_list.append([k, v.permute(0, 2, 3, 1).contiguous()])
        pred_dict = OrderedDict(pred_list)

        # Softmax the nuclear pixel map
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]

        # Softmax the type map
        type_map = F.softmax(pred_dict["tp"], dim=-1)
        type_map = torch.argmax(type_map, dim=-1, keepdim=True)
        type_map = type_map.type(torch.float32)
        pred_dict["tp"] = type_map
        pred_output = torch.cat(list(pred_dict.values()), -1)

    return pred_output.cpu().numpy()
