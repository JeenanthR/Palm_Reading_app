"""LLM provider layer. 'gemini' uses the Gemini API; 'mock' is an offline deterministic fallback
so the whole app can be demonstrated/tested without an API key."""
from . import config as C


class LLMError(Exception):
    pass


def generate(system: str, prompt: str, json_mode: bool = False) -> str:
    if C.LLM_PROVIDER == "gemini":
        if not C.GEMINI_API_KEY:
            raise LLMError("GEMINI_API_KEY is not set. Add it to .env or set LLM_PROVIDER=mock.")
        from google import genai
        from google.genai import types
        client = genai.Client(vertexai=True, api_key=C.GEMINI_API_KEY)
        cfg = types.GenerateContentConfig(system_instruction=system, temperature=0.4,
                                          response_mime_type="application/json" if json_mode else "text/plain")
        try:
            r = client.models.generate_content(model=C.GEMINI_MODEL, contents=prompt, config=cfg)
            return r.text or ""
        except Exception as e:                                # network / quota / model name errors
            raise LLMError(f"Gemini request failed: {e}") from e
    raise LLMError("mock")                                    # caller switches to template generation
