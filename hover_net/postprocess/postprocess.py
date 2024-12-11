# Packages
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, measurements
from skimage.segmentation import watershed

# HoVer Net
from hover_net.dataloader.preprocessing import get_bounding_box

def __proc_np_hv(pred):
    """
    Process Nuclei Prediction with XY Coordinate Map.

    Args:
        - pred: np.array(H, W, C)
            C=0: nuclear pixel map,
            C=1: horizontal map,
            C=2: vertical map
    Returns:
        - proced_pred: np.array(H, W)
            Numbered map of all nuclear instances.
    
    Source: https://github.com/vqdang/hover_net
    """
    pred = np.array(pred, dtype=np.float32)

    blb_raw = pred[..., 0]      # Probability map
    h_dir_raw = pred[..., 1]    # x-map
    v_dir_raw = pred[..., 2]    # y-map

    # Processing
    blb = np.array(blb_raw >= 0.5, dtype=np.int32)
    blb = measurements.label(blb)[0]
    blb[blb > 0] = 1

    # Normalize direction maps
    h_dir = cv2.normalize(
        h_dir_raw,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_dir_raw,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F
    )

    # Sobel calculates the derivaties of the image
    # The derivatives will be high when there is a high change in intensity
    # i.e. when going from one nuclei to another
    # https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html
    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    # Normalize the sobel maps
    sobelh = 1 - (
        cv2.normalize(
            sobelh,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F
        )
    )

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    # Nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = np.array(overall >= 0.4, dtype=np.int32)

    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]

    proced_pred = watershed(dist, markers=marker, mask=blb)

    return proced_pred

def process(pred_map):
    """
    Post processing of the output of the HoVer-Net model.

    Args:
        - pred_map: np.array(H, W, C)
            Combined output from all three branches of the HoVer-Net model.
            C=0: type map,
            C=1: nuclear pixel map,
            C=2: horizontal map,
            C=3: vertical map
        - nr_types (int): number of types considered at output of nc branch

    Returns:
        - pred_inst:     pixel-wise nuclear instance prediction
        - pred_type_out: dictionary containing instance information
            bbox: bounding box of the instance
            centroid: centroid of the instance
            contour: contour of the instance
            type_prob: probability of the instance belonging to a type
            type: type of the instance
    
    Based on: https://github.com/vqdang/hover_net
    """
    # Extract type and instance maps
    # pred_type: np.array(H, W, 1)
    # pred_inst: np.array(H, W, 3) => np, horizontal, vertical
    pred_type = pred_map[..., :1]
    pred_inst = pred_map[..., 1:]
    pred_type = pred_type.astype(np.int32)

    pred_inst = np.squeeze(pred_inst)
    pred_inst = __proc_np_hv(pred_inst)

    inst_info_dict = None
    # Get unique instance ids w/o background
    inst_id_list = np.unique(pred_inst)[1:]
    inst_info_dict = {}

    # Loop over each instance id
    for inst_id in inst_id_list:
        # Create map with only the current instance
        inst_map = pred_inst == inst_id

        rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
        inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
        inst_map = inst_map[
            inst_bbox[0][0]:inst_bbox[1][0],
            inst_bbox[0][1]:inst_bbox[1][1]
        ]

        inst_map = inst_map.astype(np.uint8)

        # Get the countour of the instance
        inst_contour = cv2.findContours(
            inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))

        # Skip a countour if it has less than 3 points
        # Likely an artifact
        if inst_contour.shape[0] < 3:
            continue
        if len(inst_contour.shape) != 2:
            continue

        # Get the moment of the nuclei instance
        # Moment is the "center of mass" of the instance
        # https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
        inst_moment = cv2.moments(inst_map)

        # Create centroid of the instance from the moment
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]

        inst_centroid = np.array(inst_centroid)
        inst_contour[:, 0] += inst_bbox[0][1]   # X
        inst_contour[:, 1] += inst_bbox[0][0]   # Y
        inst_centroid[0] += inst_bbox[0][1]     # X
        inst_centroid[1] += inst_bbox[0][0]     # Y
        inst_info_dict[inst_id] = {             # inst_id should start at 1
            "bbox": inst_bbox,
            "centroid": inst_centroid,
            "contour": inst_contour,
            "type_prob": None,
            "type": None,
        }

    # * Get class of each instance id, stored at index id-1
    for inst_id in list(inst_info_dict.keys()):
        rmin, cmin, rmax, cmax = (
            inst_info_dict[inst_id]["bbox"]
        ).flatten()
        inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
        inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
        inst_map_crop = (
            inst_map_crop == inst_id
        )
        inst_type = inst_type_crop[inst_map_crop]
        type_list, type_pixels = np.unique(inst_type, return_counts=True)
        type_list = list(zip(type_list, type_pixels))
        type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
        inst_type = type_list[0][0]
        if inst_type == 0:  # ! pick the 2nd most dominant if exist
            if len(type_list) > 1:
                inst_type = type_list[1][0]
        type_dict = {v[0]: v[1] for v in type_list}
        type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
        inst_info_dict[inst_id]["type"] = int(inst_type)
        inst_info_dict[inst_id]["type_prob"] = float(type_prob)

    return pred_inst, inst_info_dict