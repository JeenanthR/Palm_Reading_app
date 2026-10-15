"""Run:  python -m pytest -q   (or  python tests/test_pipeline.py)
Synthetic 'palm' image so tests need no personal photos. Real-photo tests: see docs/TEST_PLAN.md."""
import os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LLM_PROVIDER"] = "mock"
from core.validation import validate_upload
from core.palm_features import analyse_palm, locate_hand
from core.report import generate_report
from core.chatbot import PalmChatbot


def synthetic_palm(w=900, h=1100):
    img = np.full((h, w, 3), (120, 110, 100), np.uint8)
    skin = (140, 170, 215)                                            # BGR skin tone
    cv2.rectangle(img, (230, 450), (700, 1000), skin, -1)            # palm
    for i, x in enumerate([260, 380, 500, 620]):                     # fingers
        cv2.rectangle(img, (x, int(120 + abs(i - 1.5) * 40)), (x + 80, 470), skin, -1)
    cv2.ellipse(img, (210, 760), (70, 140), 0, 0, 360, skin, -1)      # thumb mound
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    dark = (90, 115, 160)
    cv2.polylines(img, [np.array([[640, 540], [520, 560], [400, 570], [300, 540]])], False, dark, 4)  # heart
    cv2.polylines(img, [np.array([[250, 600], [400, 640], [520, 660], [640, 650]])], False, dark, 4)  # head
    cv2.polylines(img, [np.array([[300, 560], [270, 700], [290, 820], [380, 940]])], False, dark, 4)  # life
    return img


def enc(img, ext=".jpg"):
    return cv2.imencode(ext, img)[1].tobytes()


def test_rejects_bad_files():
    assert not validate_upload(b"hello", "x.gif").ok
    assert not validate_upload(b"not an image", "x.jpg").ok
    assert not validate_upload(enc(np.zeros((100, 100, 3), np.uint8)), "tiny.jpg").ok
    blurry = cv2.GaussianBlur(synthetic_palm(), (0, 0), 25)
    assert not validate_upload(enc(blurry), "blur.jpg").ok


def test_full_pipeline():
    img = synthetic_palm()
    r = validate_upload(enc(img), "ok.jpg", locate_hand, "right")
    assert r.ok, r.errors
    a_l, a_r = analyse_palm(img, "left"), analyse_palm(img, "right")
    feats = {"left": a_l["features"], "right": a_r["features"]}
    rep = generate_report(feats)
    assert "NOT a scientifically validated" in rep["markdown"]
    bot = PalmChatbot(feats, rep["markdown"])
    assert "heart" in bot.ask("What does my heart line traditionally represent?").lower()
    assert len(bot.history) == 2
    bot.ask("And my career?")
    assert len(bot.history) == 4


if __name__ == "__main__":
    test_rejects_bad_files(); test_full_pipeline(); print("all tests passed")
