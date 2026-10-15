"""Task 02 - digital image processing pipeline (>= 4 techniques, each stage kept for display)."""
import cv2
import numpy as np
from skimage.filters import sato

from . import config as C


def resize_normalise(bgr: np.ndarray) -> np.ndarray:
    """Technique 1: resize (keep aspect ratio, longest side = WORK_SIZE)."""
    h, w = bgr.shape[:2]
    s = C.WORK_SIZE / max(h, w)
    return cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def preprocess_canonical(canon_bgr: np.ndarray, canon_mask: np.ndarray) -> dict:
    """Run the enhancement chain on the canonical (aligned) palm image.

    Returns a dict of named stages (all uint8 images) used both for line detection
    and for the original-vs-processed comparison in the UI.
    """
    stages = {}
    stages["original"] = canon_bgr
    # Technique 2: RGB -> grayscale
    gray = cv2.cvtColor(canon_bgr, cv2.COLOR_BGR2GRAY)
    stages["grayscale"] = gray
    # Technique 3: Gaussian smoothing (noise reduction)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    stages["gaussian"] = blur
    # Technique 4: contrast enhancement - global histogram equalisation (for comparison) and CLAHE (used)
    stages["hist_eq"] = cv2.equalizeHist(blur)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(blur)
    stages["clahe"] = clahe
    # Technique 5: edge detection - Canny and Sobel magnitude
    med = float(np.median(clahe[canon_mask > 0])) if canon_mask.any() else float(np.median(clahe))
    lo, hi = int(max(0, 0.66 * med)), int(min(255, 1.33 * med))
    stages["canny"] = cv2.bitwise_and(cv2.Canny(clahe, lo, hi), canon_mask)
    gx = cv2.Sobel(clahe, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(clahe, cv2.CV_32F, 0, 1, ksize=3)
    sob = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    stages["sobel"] = cv2.bitwise_and(sob, canon_mask)
    # Technique 6: ridge/vesselness filter (Sato) - palm creases are dark, thin, line-like ridges
    ridge = sato(clahe.astype(np.float32) / 255.0, sigmas=[1.5, 2.0, 3.0], black_ridges=True)
    ridge = ridge * (canon_mask > 0)
    ridge_u8 = cv2.normalize(ridge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    stages["ridge"] = ridge_u8
    # Technique 7: thresholding / segmentation (percentile threshold inside palm only)
    vals = ridge_u8[canon_mask > 0]
    thr = np.percentile(vals, 88) if vals.size else 128
    binary = ((ridge_u8 >= thr) & (canon_mask > 0)).astype(np.uint8) * 255
    stages["threshold"] = binary
    # Technique 8: morphological opening/closing - remove specks, bridge small gaps in creases
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    stages["morphology"] = morph
    return stages
