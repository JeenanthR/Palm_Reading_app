"""Safeguards against unsupported claims (documented for the technical report)."""
import re
from . import config as C

# Consequential / medical / absolute claims that must never be presented as fact.
BANNED = [r"\byou (will|are going to) (die|get cancer|go bankrupt|get divorced)\b", r"\bdiagnos\w*",
          r"\bcancer\b", r"\bheart (attack|disease)\b", r"\blifespan\b.*\bwill\b", r"\bguarantee\w*",
          r"\binvest in\b", r"\bquit your job\b", r"\bleave (him|her|your partner)\b"]


def sanitise(text: str) -> str:
    for pat in BANNED:
        text = re.sub(pat, "[removed: unsupported claim]", text, flags=re.I)
    return text


def ensure_disclaimer(text: str) -> str:
    if "NOT a scientifically validated" not in text:
        text = text.rstrip() + "\n\n_" + C.DISCLAIMER + "_"
    return text


def check_grounding(text: str, features_by_hand: dict) -> list[str]:
    """Flag statements about lines the CV stage did NOT detect, if they are stated as if visible."""
    issues = []
    for hand, f in features_by_hand.items():
        for ln in f["lines"]:
            if not ln["detected"]:
                nice = ln["name"].replace("_", " ")
                pat = rf"your {hand} (palm )?(has|shows) (a |an )?(clear |deep |long )?{nice}"
                if re.search(pat, text, flags=re.I):
                    issues.append(f"Claimed a visible {nice} on the {hand} palm, but it was not detected.")
    return issues
