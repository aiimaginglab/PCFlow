import cv2
import numpy as np

import pyiqa
from tqdm import tqdm

from basicsr.metrics import calculate_niqe as calculate_niqe_basicsr


def compute_niqe(restored_paths, device="cuda:0"):
    """
    Compute NIQE using basicsr.
    """
    niqe_vals = []
    for img_path in tqdm(restored_paths, desc="  NIQE"):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        niqe_score = calculate_niqe_basicsr(
            img, crop_border=0, input_order="HWC", convert_to="y"
        )
        niqe_vals.append(niqe_score)

    niqe_mean = float(np.mean(niqe_vals)) if niqe_vals else float("nan")
    return {"NIQE": niqe_mean}

def compute_musiq(restored_paths, device="cuda:0"):
    """
    Compute MUSIQ using pyiqa.
    """
    metric = pyiqa.create_metric("musiq", device=device)
    scores = []
    for path in tqdm(restored_paths, desc="MUSIQ"):
        score = metric(path).item()
        scores.append(score)
    return {"MUSIQ": float(np.mean(scores))}