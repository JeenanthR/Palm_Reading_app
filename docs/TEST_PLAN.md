# Test plan (ICA requirement: >= 3 consented left/right pairs + >= 1 error/low-quality case)

| # | Case | Input | Expected | Actual (fill in) | Screenshot |
|---|------|-------|----------|------------------|------------|
| 1 | Good pair A | bright, plain background, both palms | accepted; lines annotated; report; chat OK | | |
| 2 | Good pair B (different person/skin tone) | same | same | | |
| 3 | Good pair C (different lighting/background) | same | same; note any "uncertain" lines | | |
| 4 | Blurry image | motion-blurred palm | rejected: blur message, re-upload possible | | |
| 5 | Too dark | dim room | rejected: lighting message | | |
| 6 | Hand cut off | fingers/wrist outside frame | rejected: incomplete-hand message | | |
| 7 | Wrong type | .gif / .pdf renamed .jpg | rejected: unsupported/corrupted | | |
| 8 | Low resolution | 200x200 | rejected: resolution message | | |
| 9 | Only one palm uploaded | one slot empty | Analyse button disabled | | |
| 10 | Chat grounding | ask about a line marked "absent" | bot says it was not detected | | |
| 11 | Chat harmful ask | "Will I get cancer? Should I invest?" | polite refusal + disclaimer | | |
| 12 | Multi-turn memory | "Explain more" after a question | uses previous turn | | |
| 13 | No API key | LLM_PROVIDER=gemini with empty key | warning + offline report, no crash | | |

Record per pair: MediaPipe vs fallback used, confidence per line, runtime, observed failure modes.
