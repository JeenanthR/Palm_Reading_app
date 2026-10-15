"""Task 05 - context-aware chatbot grounded in CV features + the generated report + RAG excerpts."""
import json
from . import config as C
from .guardrails import sanitise, ensure_disclaimer
from .llm import generate, LLMError
from .rag import PalmistryKB

CHAT_SYSTEM = f"""You are the follow-up assistant of an educational palm-reading application.
GROUNDING RULES:
- Answer ONLY from the CV FEATURES, the REPORT and the KNOWLEDGE excerpts given. If the user asks about a feature
  that is 'absent' or not in the data, say it was not detected - never invent unseen palm features.
- Distinguish "what the image analysis observed" from "what palmistry traditionally says".
- Palmistry is entertainment/cultural, not science: never give predictions, medical, financial, job or relationship
  advice; decline such requests politely and restate that this is for fun.
- Keep answers short and clear (<= 150 words) and use earlier turns of the conversation for context.
- Mention the entertainment nature briefly in every answer.
Disclaimer: {C.DISCLAIMER}"""


class PalmChatbot:
    def __init__(self, features_by_hand: dict, report_md: str, kb: PalmistryKB | None = None):
        self.features, self.report, self.kb = features_by_hand, report_md, kb or PalmistryKB()
        self.history: list[tuple[str, str]] = []

    def ask(self, question: str) -> str:
        chunks = self.kb.retrieve(question, k=4)
        kb_txt = "\n".join(f"- [{c['title']}] {c['text']}" for c in chunks)
        turns = "\n".join(f"{r.upper()}: {t}" for r, t in self.history[-8:])     # keep last 8 turns
        prompt = (f"CV FEATURES:\n{json.dumps(self.features)}\n\nREPORT:\n{self.report}\n\nKNOWLEDGE:\n{kb_txt}\n\n"
                  f"CONVERSATION SO FAR:\n{turns}\n\nUSER: {question}\nASSISTANT:")
        try:
            ans = generate(CHAT_SYSTEM, prompt)
        except LLMError as e:
            if str(e) != "mock":
                ans = f"(LLM unavailable: {e})\n\n" + self._offline(question, chunks)
            else:
                ans = self._offline(question, chunks)
        ans = ensure_disclaimer(sanitise(ans)) if len(ans) < 20 else sanitise(ans)
        if "entertainment" not in ans.lower() and "not a scientifically" not in ans.lower():
            ans += "\n\n_(For entertainment only - palmistry is not scientifically validated.)_"
        self.history += [("user", question), ("assistant", ans)]
        return ans

    def _offline(self, q: str, chunks) -> str:
        ql = q.lower()
        obs = []
        for hand, f in self.features.items():
            for ln in f["lines"]:
                nm = ln["name"].replace("_", " ")
                if any(w in ql for w in nm.split()) or nm.split()[0] in ql:
                    obs.append(f"On your {hand} palm the {nm} was {ln['status']}"
                               + ("" if ln["status"] == "absent" else f" (confidence {ln['confidence']}, {ln['depth']})") + ".")
        tradition = chunks[0]["text"] if chunks else "I don't have a traditional reading for that in my knowledge base."
        return ("**Observed by image analysis:** " + (" ".join(obs) or "nothing specific to this question.") +
                "\n\n**Traditionally:** " + tradition)
