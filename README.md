# AI-Powered Palm Reading & Chatbot (Entertainment Only)

NIBM KIC - Digital Image Processing - ICA 01. Python + OpenCV + MediaPipe + RAG-grounded LLM (Gemini) + Streamlit.

> Palmistry is a cultural/entertainment practice, not scientifically validated. All output is labelled accordingly.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # use Python 3.10-3.12 (mediapipe 0.10.14)
cp .env.example .env                 # Windows: copy .env.example .env
```
Edit `.env`:
| Variable | Meaning |
|---|---|
| `LLM_PROVIDER` | `gemini` (real LLM) or `mock` (offline template, no key - good for first test) |
| `GEMINI_API_KEY` | Free key from Google AI Studio. Never commit `.env`. |
| `GEMINI_MODEL` | Model id (default `gemini-2.5-flash`; change if your account lists another) |

## Run
```bash
streamlit run app.py
python tests/test_pipeline.py        # automated checks (synthetic image, mock LLM)
```
Workflow in the UI: upload left + right palm -> validation messages -> Analyse -> tabs (processing, features, report, chatbot, evidence).

## Project layout
```
app.py                    Streamlit UI (Tasks 01, 06)
core/validation.py        Task 01  format / corruption / resolution / blur / lighting / hand check
core/preprocessing.py     Task 02  resize, grayscale, Gaussian, hist-eq, CLAHE, Canny, Sobel, Sato ridge, threshold, morphology
core/palm_features.py     Task 03  MediaPipe landmarks, palm alignment, line detection, annotation, palm shape
core/rag.py               Task 04  TF-IDF retrieval over knowledge_base/palmistry.md
core/report.py            Task 04  structured report (LLM JSON) + offline fallback
core/guardrails.py        Task 04/05 unsupported-claim safeguards
core/chatbot.py           Task 05  grounded multi-turn chatbot
knowledge_base/palmistry.md  curated palmistry knowledge (edit to extend)
docs/GUIDE.md             step-by-step build guide, viva Q&A, report outline
docs/TEST_PLAN.md         test cases + results table
```

## Mediapipe note
`mediapipe==0.10.14` has the bundled `mp.solutions.hands` model. If you use a newer mediapipe, download
`hand_landmarker.task` from the MediaPipe docs into `models/`. If MediaPipe is unavailable the app falls back to
skin-colour segmentation and clearly marks results as lower confidence.

## Privacy
Images are processed in memory only and never written to disk or sent to the LLM (only extracted feature JSON is sent).
Use photos only with the owner's consent. API keys come from environment variables only.

## Third-party components / AI assistance (acknowledge in your report)
OpenCV, NumPy, scikit-image, scikit-learn, MediaPipe, Streamlit, Google Gemini API, Pillow. Code scaffold generated with AI assistance (Claude) and reviewed/tested by the student.
