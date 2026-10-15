"""Central configuration. Secrets come ONLY from environment variables (.env)."""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_MB = 10
MIN_SIDE_PX = 400            # reject images smaller than this
BLUR_REJECT = 40.0           # variance of Laplacian below this -> reject
BLUR_WARN = 100.0            # below this -> warn
DARK_MEAN = 60               # mean gray below -> too dark
BRIGHT_MEAN = 210            # mean gray above -> overexposed
MIN_CONTRAST_STD = 25        # gray std below -> flat / low contrast
WORK_SIZE = 800              # longest side after resize (normalisation step)
CANON = 400                  # canonical palm image is CANON x CANON

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

DISCLAIMER = (
    "Entertainment / cultural notice: palmistry is a traditional practice and is NOT a "
    "scientifically validated way to predict personality, health, wealth, relationships "
    "or future events. Do not use this output for medical, financial, employment, "
    "relationship or other important decisions."
)
