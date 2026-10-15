"""Task 03 - locate the hand, align the palm into a canonical frame and detect crease lines.

Visual (computer-vision) observations are returned with source="cv" and a confidence value;
nothing here is an interpretation. Interpretation is done later by the LLM and is labelled separately.

Canonical frame (400x400, thumb ALWAYS on the left, fingers at the top):
    index MCP -> (60, 40)   pinky MCP -> (340, 40)   wrist centre -> (200, 380)
"""
from dataclasses import dataclass, field
import os
import cv2
import numpy as np
from skimage.morphology import skeletonize

from . import config as C
from .preprocessing import preprocess_canonical

S = C.CANON
CANON_PTS = np.float32([[60, 40], [340, 40], [200, 380]])    # index MCP, pinky MCP, wrist

# Search regions (x0, y0, x1, y1) in the canonical frame + expected orientation (degrees from horizontal)
REGIONS = {
    "heart_line": dict(box=(90, 60, 350, 175), orient="horizontal"),
    "head_line":  dict(box=(40, 140, 330, 260), orient="horizontal"),
    "life_line":  dict(box=(20, 130, 230, 390), orient="vertical"),
    "fate_line":  dict(box=(150, 150, 270, 390), orient="vertical"),
}
COLORS = {"heart_line": (60, 60, 255), "head_line": (255, 160, 40), "life_line": (60, 200, 60),
          "fate_line": (200, 60, 200)}


@dataclass
class HandLocation:
    found: bool
    method: str = "none"
    landmarks: np.ndarray | None = None      # (21,2) pixel coords (MediaPipe order) or pseudo-landmarks
    mask: np.ndarray | None = None           # palm ROI mask in original image space
    touches_border: bool = False
    pseudo: bool = False
    notes: list = field(default_factory=list)


# ----------------------------------------------------------------------------- hand localisation
_mp_hands = None
_mp_tasks = None


def _mediapipe_landmarks(bgr):
    """Return (21,2) pixel landmarks or None. Supports legacy solutions API and Tasks API."""
    global _mp_hands, _mp_tasks
    try:
        import mediapipe as mp
    except Exception:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]
    try:
        if hasattr(mp, "solutions"):
            if _mp_hands is None:
                _mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                                     min_detection_confidence=0.5)
            r = _mp_hands.process(rgb)
            if r.multi_hand_landmarks:
                lm = r.multi_hand_landmarks[0].landmark
                return np.array([[p.x * w, p.y * h] for p in lm], np.float32)
        else:                                  # newer mediapipe: Tasks API needs models/hand_landmarker.task
            model = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")
            if not os.path.exists(model):
                return None
            from mediapipe.tasks import python as mpt
            from mediapipe.tasks.python import vision
            if _mp_tasks is None:
                opts = vision.HandLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=model),
                                                    num_hands=1)
                _mp_tasks = vision.HandLandmarker.create_from_options(opts)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            r = _mp_tasks.detect(img)
            if r.hand_landmarks:
                return np.array([[p.x * w, p.y * h] for p in r.hand_landmarks[0]], np.float32)
    except Exception:
        return None
    return None


def _skin_mask(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycc, (0, 133, 77), (255, 180, 135))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return m


def locate_hand(bgr: np.ndarray, hand_label: str = "right") -> HandLocation:
    h, w = bgr.shape[:2]
    lm = _mediapipe_landmarks(bgr)
    if lm is not None:
        poly = cv2.convexHull(lm[[0, 1, 5, 9, 13, 17]].astype(np.int32))
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, poly, 255)
        margin = 0.01 * min(h, w)
        tb = bool((lm[:, 0].min() < margin) or (lm[:, 1].min() < margin) or
                  (lm[:, 0].max() > w - margin) or (lm[:, 1].max() > h - margin))
        return HandLocation(True, "mediapipe", lm, mask, tb)

    # ---- fallback: skin segmentation + pseudo-landmarks (less accurate -> flagged) ----
    skin = _skin_mask(bgr)
    cnts, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return HandLocation(False)
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.05 * h * w:
        return HandLocation(False)
    x, y, bw, bh = cv2.boundingRect(c)
    touches = bool(x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2)
    # Assume fingers up. Thumb on image-right for a RIGHT palm facing the camera, image-left for LEFT.
    thumb_right = (hand_label == "right")
    top = y + 0.42 * bh
    xi, xp = (x + 0.80 * bw, x + 0.20 * bw) if thumb_right else (x + 0.20 * bw, x + 0.80 * bw)
    lm = np.zeros((21, 2), np.float32)
    lm[0] = (x + 0.5 * bw, y + 0.98 * bh)           # wrist
    lm[5] = (xi, top)                                # index MCP
    lm[17] = (xp, top)                               # pinky MCP
    lm[9] = (x + 0.5 * bw, top)                      # middle MCP
    lm[13] = (x + 0.5 * bw, top)
    lm[1] = (xi, y + 0.80 * bh)
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [c], -1, 255, -1)
    return HandLocation(True, "skin-segmentation-fallback", lm, mask, touches, True,
                        ["MediaPipe was unavailable or found no hand: using the less accurate skin-colour fallback. "
                         "Palm alignment is approximate, so line locations are lower-confidence."])


# ----------------------------------------------------------------------------- canonical alignment
def align_palm(bgr: np.ndarray, loc: HandLocation):
    """Warp the palm to the canonical frame. Returns (canon_bgr, canon_mask, M, Minv)."""
    src = loc.landmarks[[5, 17, 0]].astype(np.float32)
    M = cv2.getAffineTransform(src, CANON_PTS)
    canon = cv2.warpAffine(bgr, M, (S, S), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # palm mask: trapezoid in canonical space (index/pinky MCP -> wrist, widened a little for the thumb mound)
    poly = np.array([[20, 40], [350, 40], [310, 385], [90, 385]], np.int32)
    palm = np.zeros((S, S), np.uint8)
    cv2.fillConvexPoly(palm, poly, 255)
    # intersect with the real hand mask (warped) so background is excluded
    wm = cv2.warpAffine(loc.mask, M, (S, S), flags=cv2.INTER_NEAREST)
    mask = cv2.bitwise_and(palm, wm) if loc.method == "mediapipe" else cv2.bitwise_and(palm, wm)
    if mask.sum() == 0:
        mask = palm
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))   # stay away from the palm boundary (false edges)
    return canon, mask, M, cv2.invertAffineTransform(M)


# ----------------------------------------------------------------------------- line detection
def _best_component(skel, region, orient, ridge):
    x0, y0, x1, y1 = region
    box = np.zeros_like(skel, np.uint8)
    box[y0:y1, x0:x1] = 1
    sk = (skel > 0).astype(np.uint8) * box
    n, lab, stats, _ = cv2.connectedComponentsWithStats(sk, connectivity=8)
    best, best_score, extra = None, 0.0, 0
    for i in range(1, n):
        ys, xs = np.where(lab == i)
        if len(xs) < 25:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        mean, vecs = cv2.PCACompute(pts, mean=None, maxComponents=1)
        ang = abs(np.degrees(np.arctan2(vecs[0, 1], vecs[0, 0])))
        ang = min(ang, 180 - ang)                         # 0 = horizontal, 90 = vertical
        orient_ok = (1 - ang / 90.0) if orient == "horizontal" else (ang / 90.0)
        strength = float(ridge[ys, xs].mean()) / 255.0
        score = len(xs) * (0.4 + 0.6 * orient_ok) * (0.5 + strength)
        if len(xs) >= 40:
            extra += 1
        if score > best_score:
            best, best_score = (pts, strength, orient_ok), score
    return best, extra


def detect_lines(stages: dict, mask: np.ndarray) -> list[dict]:
    ridge, morph = stages["ridge"], stages["morphology"]
    skel = skeletonize(morph > 0).astype(np.uint8)
    results = []
    palm_len = 340.0
    for name, spec in REGIONS.items():
        best, n_comp = _best_component(skel, spec["box"], spec["orient"], ridge)
        item = {"name": name, "source": "cv", "detected": False, "status": "absent", "confidence": 0.0,
                "polyline": [], "length_ratio": 0.0, "curvature": 0.0, "depth": "n/a",
                "competing_creases_in_region": max(0, n_comp - 1)}
        if best is not None:
            pts, strength, orient_ok = best
            # order points along principal axis and smooth with a polynomial
            mean, vecs = cv2.PCACompute(pts, mean=None, maxComponents=1)
            t = (pts - mean) @ vecs[0]
            order = np.argsort(t)
            p, tt = pts[order], t[order]
            deg = 3 if len(p) > 60 else 1
            tg = np.linspace(tt.min(), tt.max(), 40)
            if abs(vecs[0, 0]) >= abs(vecs[0, 1]):       # more horizontal -> y = f(x)
                coef = np.polyfit(p[:, 0], p[:, 1], deg)
                xs = np.linspace(p[:, 0].min(), p[:, 0].max(), 40)
                poly = np.column_stack([xs, np.polyval(coef, xs)])
            else:                                         # more vertical -> x = f(y)
                coef = np.polyfit(p[:, 1], p[:, 0], deg)
                ys = np.linspace(p[:, 1].min(), p[:, 1].max(), 40)
                poly = np.column_stack([np.polyval(coef, ys), ys])
            arc = float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)))
            chord = float(np.linalg.norm(poly[-1] - poly[0])) + 1e-6
            length_ratio = arc / palm_len
            expected = {"heart_line": 0.60, "head_line": 0.60, "life_line": 0.65, "fate_line": 0.50}[name]
            len_score = min(1.0, length_ratio / expected)
            conf = float(np.clip(0.45 * len_score + 0.30 * orient_ok + 0.25 * min(1.0, strength * 2.5), 0, 1))
            if name == "fate_line":
                conf *= 0.8                                # fate line is often absent -> be conservative
            status = "detected" if conf >= 0.55 else ("uncertain" if conf >= 0.35 else "absent")
            if length_ratio < 0.25:                      # too short to be a real line -> treat as absent
                status = "absent"
            item.update(detected=status != "absent", status=status, confidence=round(conf, 2),
                        polyline=poly.round(1).tolist() if status != "absent" else [],
                        length_ratio=round(length_ratio, 2), curvature=round(arc / chord, 2),
                        depth="deep" if strength > 0.45 else ("moderate" if strength > 0.3 else "faint"))
        results.append(item)
    return results


def palm_shape(loc: HandLocation) -> dict:
    lm = loc.landmarks
    width = float(np.linalg.norm(lm[5] - lm[17]))
    length = float(np.linalg.norm(lm[0] - lm[9]))
    if loc.pseudo or length < 1e-3 or width < 1e-3:
        return {"source": "cv", "shape": "unknown", "confidence": 0.2,
                "note": "Needs MediaPipe landmarks to estimate palm proportions."}
    ratio = length / width
    finger = float(np.linalg.norm(lm[9] - lm[12])) / length
    rect = ratio >= 1.1
    longf = finger >= 0.85
    element = {(False, False): "earth-type (square palm, shorter fingers)",
               (False, True): "air-type (square palm, longer fingers)",
               (True, True): "water-type (rectangular palm, longer fingers)",
               (True, False): "fire-type (rectangular palm, shorter fingers)"}[(rect, longf)]
    return {"source": "cv", "shape": element, "palm_length_to_width": round(ratio, 2),
            "finger_to_palm_length": round(finger, 2), "confidence": 0.5,
            "note": "Estimated from 2-D landmarks; camera angle affects the ratios."}


# ----------------------------------------------------------------------------- annotation + orchestration
def annotate(canon_bgr, lines, mask):
    out = canon_bgr.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 255, 0), 1)
    for ln in lines:
        if not ln["polyline"]:
            continue
        pts = np.array(ln["polyline"], np.int32).reshape(-1, 1, 2)
        col = COLORS[ln["name"]]
        if ln["status"] == "uncertain":                       # dashed look for uncertain
            for i in range(0, len(pts) - 1, 2):
                cv2.line(out, tuple(pts[i, 0]), tuple(pts[min(i + 1, len(pts) - 1), 0]), col, 3)
        else:
            cv2.polylines(out, [pts], False, col, 3)
        label = f"{ln['name'].replace('_', ' ')} {ln['confidence']:.2f}" + ("?" if ln["status"] == "uncertain" else "")
        cv2.putText(out, label, tuple(pts[0, 0] + np.array([4, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return out


def annotate_original(bgr, loc, lines, Minv):
    """Project canonical polylines back onto the original photo."""
    out = bgr.copy()
    for ln in lines:
        if not ln["polyline"]:
            continue
        p = np.array(ln["polyline"], np.float32).reshape(-1, 1, 2)
        p = cv2.transform(p, Minv).astype(np.int32)
        cv2.polylines(out, [p], False, COLORS[ln["name"]], max(2, bgr.shape[1] // 300))
    if loc.method == "mediapipe":
        for (x, y) in loc.landmarks.astype(int):
            cv2.circle(out, (x, y), 3, (0, 255, 255), -1)
    return out


def analyse_palm(bgr: np.ndarray, hand_label: str) -> dict:
    """Full per-hand pipeline: locate -> align -> preprocess -> detect -> annotate -> structured features."""
    loc = locate_hand(bgr, hand_label)
    if not loc.found:
        raise ValueError("No hand found")
    canon, mask, M, Minv = align_palm(bgr, loc)
    stages = preprocess_canonical(canon, mask)
    lines = detect_lines(stages, mask)
    shape = palm_shape(loc)
    features = {
        "hand": hand_label,
        "localisation_method": loc.method,
        "palm_shape": shape,
        "lines": [{k: v for k, v in ln.items() if k != "polyline"} for ln in lines],
        "notes": loc.notes,
    }
    return {"features": features, "stages": stages, "mask": mask, "lines": lines,
            "annotated_canon": annotate(canon, lines, mask),
            "annotated_original": annotate_original(bgr, loc, lines, Minv), "loc": loc}
