# Core
import colorsys
import random

# Packages
import cv2
import numpy as np
import matplotlib.pyplot as plt

def proc_valid_step_output(raw_data, nr_types=None):
    """
    Calculate dice coefficient and accuracy for validation step.
    """
    # Define a dictionary to store the tracked values
    track_dict = {"scalar": {}}

    def track_value(name, value, vtype):
        return track_dict[vtype].update({name: value})

    def _dice_info(true, pred, label):
        true = np.array(true == label, np.int32)
        pred = np.array(pred == label, np.int32)
        inter = (pred * true).sum()
        total = (pred + true).sum()
        return inter, total

    # NP accuracy and dice
    over_inter = 0
    over_total = 0
    over_correct = 0
    prob_np = raw_data["prob_np"]
    true_np = raw_data["true_np"]
    for idx in range(len(raw_data["true_np"])):
        patch_prob_np = prob_np[idx]
        patch_true_np = true_np[idx]
        patch_pred_np = np.array(patch_prob_np > 0.5, dtype=np.int32)
        inter, total = _dice_info(patch_true_np, patch_pred_np, 1)
        correct = (patch_pred_np == patch_true_np).sum()
        over_inter += inter
        over_total += total
        over_correct += correct
    nr_pixels = len(true_np) * np.size(true_np[0])
    acc_np = over_correct / nr_pixels
    dice_np = 2 * over_inter / (over_total + 1.0e-8)
    track_value("np_acc", acc_np, "scalar")
    track_value("np_dice", dice_np, "scalar")

    # TP dice
    pred_tp = raw_data["pred_tp"]
    true_tp = raw_data["true_tp"]
    for type_id in range(0, nr_types):
        over_inter = 0
        over_total = 0
        for idx in range(len(raw_data["true_np"])):
            patch_pred_tp = pred_tp[idx]
            patch_true_tp = true_tp[idx]
            inter, total = _dice_info(
                patch_true_tp, patch_pred_tp, type_id
            )
            over_inter += inter
            over_total += total
        dice_tp = 2 * over_inter / (over_total + 1.0e-8)
        track_value("tp_dice_%d" % type_id, dice_tp, "scalar")

    # HV MSE
    pred_hv = raw_data["pred_hv"]
    true_hv = raw_data["true_hv"]
    over_squared_error = 0
    for idx in range(len(raw_data["true_np"])):
        patch_pred_hv = pred_hv[idx]
        patch_true_hv = true_hv[idx]
        squared_error = patch_pred_hv - patch_true_hv
        squared_error = squared_error * squared_error
        over_squared_error += squared_error.sum()
    mse = over_squared_error / nr_pixels
    track_value("hv_mse", mse, "scalar")

    return track_dict