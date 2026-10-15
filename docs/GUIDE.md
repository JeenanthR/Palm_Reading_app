# Start-to-end guide: AI Palm Reading & Chatbot (ICA 01)

## 0. Plan your time (suggested)
| Day | Goal | Marks covered |
|---|---|---|
| 1 | Environment, run the starter app, collect 3 consented palm pairs | - |
| 2 | Understand + tune validation and preprocessing (Tasks 01-02) | 30 |
| 3 | Tune feature detection and annotations (Task 03) | 20 |
| 4 | Add Gemini key, test RAG report + chatbot (Tasks 04-05) | 35 |
| 5 | Integration, error cases, screenshots (Task 06) | 5 |
| 6 | Technical report (5-8 pages) + viva rehearsal | 10 |

## 1. Architecture
```
Upload L/R -> Validate -> Locate hand (MediaPipe) -> Align palm to canonical 400x400 frame
 -> Preprocess (gray, Gaussian, CLAHE, Canny/Sobel, Sato ridge, threshold, morphology)
 -> Detect + annotate lines (CV only)  ->  structured feature JSON (per hand)
 -> RAG: retrieve palmistry knowledge chunks -> Gemini -> structured report (8 sections)
 -> Guardrails -> Chatbot (features + report + retrieved chunks + history)
```
Key design idea: **observation (CV) is separated from interpretation (LLM)**. Features carry `source: "cv"`,
`status` (detected / uncertain / absent) and `confidence`; the LLM may only interpret them.

## 2. Step-by-step build
1. **Environment**: follow README. First run with `LLM_PROVIDER=mock` to prove the CV pipeline works.
2. **Task 01 (`validation.py`)** - checks, in order: extension -> empty/size -> PIL `verify()` (corruption/spoofed extension)
   -> decode -> resolution -> blur (variance of Laplacian) -> brightness/contrast -> hand present -> hand touches border.
   Every failure returns an *actionable* message; Streamlit uploader lets the user replace the file.
3. **Task 02 (`preprocessing.py`)** - eight techniques, all shown in the UI for original-vs-processed comparison:
   resize/normalise, grayscale, Gaussian, histogram equalisation, CLAHE, Canny, Sobel, Sato ridge filter,
   thresholding, morphology (open then close).
4. **Task 03 (`palm_features.py`)**:
   - MediaPipe gives 21 hand landmarks -> palm polygon.
   - An affine warp maps index-MCP, pinky-MCP and wrist to fixed canonical points, so thumb is always on the left and the
     palm is a fixed 400x400 frame (works for both hands and any rotation).
   - Lines are searched in fixed anatomical regions of that frame (heart: upper band; head: middle band; life: arc around
     thumb mound; fate: central vertical). Skeletonise the cleaned ridge map, take connected components, score by length x
     orientation x ridge strength, fit a cubic polynomial, compute length ratio / curvature / depth, assign confidence.
   - Status rules: confidence >= 0.55 detected, 0.35-0.55 uncertain (dashed in the picture), < 0.35 or too short = absent.
   - Palm shape (earth/air/water/fire) from landmark proportions - flagged low confidence.
5. **Task 04 (`rag.py`, `report.py`)** - chosen specialisation: **general LLM + curated knowledge base via RAG**
   (allowed option 3). Document in the report: model = Gemini (name from `.env`); KB = `palmistry.md` (~25 chunks);
   retrieval = TF-IDF cosine, top-10 for the report, top-4 per chat question; prompt = system rules + feature JSON +
   retrieved chunks; output = JSON with 8 sections; safeguards = prompt rules, `sanitise()` regex blacklist,
   `check_grounding()`, forced disclaimer, "absent" features never described as visible.
   *Improve the mark*: add more chunks to `palmistry.md` from books/websites you cite; optionally swap TF-IDF for
   embeddings (e.g. sentence-transformers) and compare.
6. **Task 05 (`chatbot.py`)** - every question gets: features JSON + report + fresh retrieval + last 8 turns.
   Refuses medical/financial/predictive asks, never invents unseen features, repeats the disclaimer.
7. **Task 06 (`app.py`)** - single Streamlit app; no manual scripts. Analyse button disabled until both palms are valid.

## 3. How to photograph good test palms (this decides your detection quality)
Open palm facing camera, fingers slightly apart, plain dark/neutral background, soft daylight, no flash, camera ~30 cm
away and parallel to the palm, whole hand + wrist inside the frame, tap-to-focus on the palm. Get **consent** from everyone.

## 4. Tuning checklist (do this with your real photos)
- If lines are mis-assigned: adjust `REGIONS` boxes and `CANON_PTS` in `palm_features.py`.
- Too many/too few lines: change the percentile (88) in the threshold stage, Sato sigmas, morphology kernels.
- Blur/lighting rejections too strict/lenient: edit thresholds in `core/config.py` and note the values you chose.
- Keep a table of parameters tried and results - it becomes your "test findings" section.

## 5. Honest limitations to write in the report
Heuristic regions assume a flat, front-facing palm; skin texture and lighting create false creases; fate line is often
absent; palm-shape ratios are distorted by camera angle; TF-IDF retrieval is keyword-based; the LLM can still be
imperfect even with guardrails; palmistry itself is not validated. (The marks reward honesty about this.)

## 6. Technical report outline (5-8 pages)
1 Introduction + disclaimer · 2 System architecture (diagram above) · 3 Validation · 4 Preprocessing (each technique:
what, why, parameters, before/after figure) · 5 Feature extraction (alignment, regions, scoring, confidence) ·
6 LLM specialisation (model, KB, retrieval, prompts, I/O, safeguards) · 7 Chatbot design · 8 Testing (use TEST_PLAN
table, incl. error cases) · 9 Limitations & ethics/privacy · 10 Third-party/AI acknowledgement · References.

## 7. Live demo script (5-7 min)
1. Show disclaimer. 2. Upload a bad file (error message) then a good left + right palm. 3. Show processing tab and
explain each stage. 4. Show annotated lines, point out one "uncertain" and one "absent". 5. Show report sections and
retrieved chunks. 6. Chat: heart line -> left/right differences -> career -> follow-up "explain more" -> a harmful
question (refusal). 7. Mention privacy (no disk storage, key in `.env`).

## 8. Viva - questions you must be able to answer
- Why CLAHE instead of plain histogram equalisation? (local contrast, avoids amplifying noise/over-brightening)
- Why Gaussian before Canny/ridge? (suppress noise, avoid false edges) Why Canny **and** Sobel? (compare; Canny thin/thresholded, Sobel gradient strength)
- What is the Sato filter? (Hessian-based multi-scale ridge detector for thin dark curves such as creases)
- Why morphology? (opening removes specks; closing bridges small gaps)
- Why warp to a canonical frame? (position-independent regions, same code for left/right, scale/rotation invariance)
- How is confidence computed? (length vs expected, orientation match, ridge strength)
- Why RAG instead of fine-tuning? (no training data/compute, transparent, editable knowledge, traceable chunks)
- How do you stop hallucination? (grounding rules, absent-feature handling, sanitiser, forced disclaimer, features-only inputs)
- What happens if MediaPipe fails? (skin-segmentation fallback with warning)
- Privacy? (in-memory, only feature JSON sent, env-var keys)

## 9. Submission checklist
Source code zipped (no `.env`) · README · 5-8 page report · screenshots (uploads, original/processed, annotated,
report, chat) · TEST_PLAN filled with >= 3 pairs + error case · acknowledgements of third-party/AI components.
