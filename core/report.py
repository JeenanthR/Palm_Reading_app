"""Task 04 - structured report generation (RAG-grounded LLM) with an offline template fallback."""
import json
import re

from . import config as C
from .guardrails import sanitise, ensure_disclaimer, check_grounding
from .llm import generate, LLMError
from .rag import PalmistryKB, features_to_query

SECTIONS = ["major_lines", "personality", "career", "finance", "relationships", "future",
            "left_vs_right", "uncertainty"]
TITLES = {"major_lines": "Major lines & visible characteristics", "personality": "Personality themes",
          "career": "Career & professional-life themes", "finance": "Financial & prosperity themes",
          "relationships": "Relationships & emotional themes", "future": "General future-oriented narrative",
          "left_vs_right": "Left vs right palm comparison", "uncertainty": "Uncertainty & limits"}

SYSTEM_PROMPT = f"""You are a palmistry-knowledgeable assistant for an educational computer-vision project.
RULES (never break them):
1. Palmistry is traditional/entertainment only. Never present anything as a fact or prediction. Use phrases like
   "traditionally read as", "palmistry suggests", "symbolically".
2. Base statements ONLY on (a) the CV FEATURES JSON and (b) the KNOWLEDGE excerpts supplied. Features with
   status "absent" were not seen: do not describe them as visible; say they were not detected.
   Features with status "uncertain" must be called uncertain.
3. Keep OBSERVATIONS (what the image analysis found) clearly separate from INTERPRETATION (traditional meaning).
4. No medical, legal, financial, employment or relationship advice or diagnoses. No lifespan/health/death claims.
5. Be kind, concise and balanced. Always include the entertainment disclaimer.
Disclaimer: {C.DISCLAIMER}"""


def _features_text(fb):
    return json.dumps(fb, indent=1)


def build_prompt(features_by_hand, chunks):
    kb = "\n".join(f"- [{c['title']}] {c['text']}" for c in chunks)
    return f"""CV FEATURES (computer-vision output, per hand):
{_features_text(features_by_hand)}

KNOWLEDGE (curated palmistry excerpts, retrieved with RAG):
{kb}

Write the reading as JSON with exactly these string keys: {SECTIONS}.
Each value: 2-5 sentences (markdown allowed). In "major_lines" start each line with "Observation:" then "Traditional reading:".
In "left_vs_right" compare the two hands using the feature differences and the left/right tradition.
In "uncertainty" list which detections were uncertain/absent and why photos limit accuracy."""


def _mock_report(fb, chunks):
    """Offline, deterministic report built directly from features + retrieved chunks."""
    by_title = {c["title"]: c["text"] for c in chunks}
    def kb(*keys):
        for k in keys:
            for t, txt in by_title.items():
                if k in t:
                    return txt
        return ""
    lines_md = []
    for hand, f in fb.items():
        for ln in f["lines"]:
            nm = ln["name"].replace("_", " ")
            if ln["status"] == "absent":
                obs = f"{hand.title()} {nm}: not detected in this photo."
            else:
                obs = (f"{hand.title()} {nm}: {ln['status']} (confidence {ln['confidence']}), relative length "
                       f"{ln['length_ratio']}, curvature {ln['curvature']}, appears {ln['depth']}.")
            lines_md.append(f"- **Observation:** {obs} **Traditional reading:** {kb(ln['name'] + ' overview')}")
    shapes = ", ".join(f"{h}: {f['palm_shape'].get('shape', 'unknown')}" for h, f in fb.items())
    diffs = []
    hands = list(fb)
    if len(hands) == 2:
        a, b = (fb[h] for h in hands)
        for la, lb in zip(a["lines"], b["lines"]):
            if la["status"] != lb["status"] or abs(la["length_ratio"] - lb["length_ratio"]) > 0.15:
                diffs.append(f"{la['name'].replace('_', ' ')} differs ({hands[0]}: {la['status']}, {la['length_ratio']}; "
                             f"{hands[1]}: {lb['status']}, {lb['length_ratio']})")
    unc = [f"{h} {l['name'].replace('_', ' ')} ({l['status']})" for h, f in fb.items() for l in f["lines"] if l["status"] != "detected"]
    return {
        "major_lines": "\n".join(lines_md),
        "personality": kb("personality themes") + f" Palm shape estimate - {shapes}.",
        "career": kb("career themes"),
        "finance": kb("financial themes"),
        "relationships": kb("relationship themes") + " " + kb("heart_line overview"),
        "future": kb("future narratives"),
        "left_vs_right": (kb("left versus right") + " Visible differences: " + ("; ".join(diffs) if diffs else "no major differences detected.")),
        "uncertainty": "Uncertain or absent detections: " + (", ".join(unc) if unc else "none") + ". " + kb("uncertainty and limits"),
    }


def generate_report(features_by_hand: dict, kb: PalmistryKB | None = None) -> dict:
    kb = kb or PalmistryKB()
    chunks = kb.retrieve(features_to_query(features_by_hand), k=10)
    mode, sections, warn = C.LLM_PROVIDER, None, []
    try:
        raw = generate(SYSTEM_PROMPT, build_prompt(features_by_hand, chunks), json_mode=True)
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        sections = json.loads(raw)
        if not all(k in sections for k in SECTIONS):
            raise ValueError("missing sections")
    except LLMError as e:
        if str(e) != "mock":
            warn.append(f"{e} - used the offline template report instead.")
        sections, mode = _mock_report(features_by_hand, chunks), "mock"
    except Exception as e:
        warn.append(f"LLM output could not be parsed ({e}) - used the offline template report instead.")
        sections, mode = _mock_report(features_by_hand, chunks), "mock"

    md = [f"# Palm reading report\n"]
    for k in SECTIONS:
        txt = sanitise(str(sections.get(k, "")))
        md.append(f"## {TITLES[k]}\n{txt}\n")
    report_md = ensure_disclaimer("\n".join(md))
    issues = check_grounding(report_md, features_by_hand)
    return {"markdown": report_md, "sections": sections, "retrieved": chunks, "mode": mode,
            "warnings": warn + issues}
