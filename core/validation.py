"""Task 01 - upload validation: format, corruption, resolution, blur, lighting, hand present."""
from dataclasses import dataclass, field
import io
import os
import cv2
import numpy as np
from PIL import Image

from . import config as C


@dataclass
class ValidationResult:
    ok: bool
    image: np.ndarray | None = None          # BGR
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def validate_upload(data: bytes, filename: str, hand_locator=None, hand_label: str = "right") -> ValidationResult:
    res = ValidationResult(ok=False)
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in C.ALLOWED_EXTENSIONS:
        res.errors.append(f"Unsupported file type '{ext}'. Please upload a JPG, JPEG or PNG image.")
        return res
    if not data:
        res.errors.append("The file is empty. Please upload again.")
        return res
    if len(data) > C.MAX_FILE_MB * 1024 * 1024:
        res.errors.append(f"File is larger than {C.MAX_FILE_MB} MB. Please upload a smaller photo.")
        return res

    # corruption / spoofed-extension check
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        res.errors.append("The file is corrupted or is not a real image. Please re-export the photo and upload again.")
        return res
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        res.errors.append("The image could not be decoded. Please try another photo.")
        return res

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean, std = float(gray.mean()), float(gray.std())
    res.metrics = {"width": w, "height": h, "blur_laplacian_var": round(lap_var, 1),
                   "mean_brightness": round(mean, 1), "contrast_std": round(std, 1)}

    if min(h, w) < C.MIN_SIDE_PX:
        res.errors.append(f"Resolution too low ({w}x{h}). Use a photo at least {C.MIN_SIDE_PX}px on the short side - move closer or use a better camera.")
    if lap_var < C.BLUR_REJECT:
        res.errors.append("The image is too blurry to see palm lines. Hold the camera steady, tap to focus on the palm and retake.")
    elif lap_var < C.BLUR_WARN:
        res.warnings.append("The image is slightly blurry; line detection may be less reliable.")
    if mean < C.DARK_MEAN:
        res.errors.append("The image is too dark. Retake in bright, even light (e.g. near a window).")
    elif mean > C.BRIGHT_MEAN:
        res.errors.append("The image is over-exposed. Avoid direct flash/sunlight on the palm and retake.")
    if std < C.MIN_CONTRAST_STD:
        res.warnings.append("Low contrast / flat lighting; creases may be hard to see.")

    if hand_locator is not None and not res.errors:
        loc = hand_locator(img, hand_label)
        res.metrics["hand_method"] = loc.method
        if not loc.found:
            res.errors.append("No hand/palm could be found. Place one open palm facing the camera against a plain background.")
        else:
            if loc.touches_border:
                res.errors.append("The hand looks cut off at the image edge (incomplete hand). Retake with the whole hand and wrist inside the frame.")
            res.warnings.extend(loc.notes)

    res.ok = not res.errors
    res.image = img if res.ok else None
    return res
