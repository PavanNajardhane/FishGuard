"""Image calibration T(I, X) with X = [B, C, G, S, Sh]  (paper eq. 1 / Sec 3.2)."""
import cv2, numpy as np
from PIL import Image, ImageEnhance

# Search space Omega for ABC: (low, high) per parameter
BOUNDS = {
    "brightness": (-50.0, 50.0),   # additive offset on 0-255 scale (paper result: 25.73)
    "contrast":   (0.6, 1.6),      # multiplicative
    "gamma":      (0.6, 1.8),
    "saturation": (0.6, 1.8),
    "sharpness":  (0.6, 2.0),
}
KEYS = list(BOUNDS.keys())

# Parameters reported in the paper (Table 4) - used as default until you run ABC yourself
PAPER_PARAMS = {"brightness": 25.72657, "contrast": 1.00721, "gamma": 1.30102,
                "saturation": 1.18816, "sharpness": 1.04718}


def vec_to_params(x):
    return {k: float(v) for k, v in zip(KEYS, x)}


def enhance_bgr(img_bgr, p):
    """Apply brightness -> contrast -> gamma -> saturation -> sharpness. Input/Output: BGR uint8."""
    img = img_bgr.astype(np.float32)
    img = img * p["contrast"] + p["brightness"]            # contrast + brightness
    img = np.clip(img, 0, 255) / 255.0
    img = np.power(img, 1.0 / max(p["gamma"], 1e-3)) * 255.0  # gamma (>1 brightens)
    img = np.clip(img, 0, 255).astype(np.uint8)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil = ImageEnhance.Color(pil).enhance(p["saturation"])
    pil = ImageEnhance.Sharpness(pil).enhance(p["sharpness"])
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
