import time

from app.services.ai.gemini import gemini_generate
from app.services.ai.openrouter import openrouter_generate
from app.utils.classifier import is_simple_query


def _try_with_retries(fn, prompt: str, retries: int = 2) -> str:
    last_exc = None
    for attempt in range(retries):
        try:
            return fn(prompt)
        except Exception as exc:
            last_exc = exc
            print(f"[ai-router] attempt {attempt + 1} failed: {exc}")
            time.sleep(0.5)
    if last_exc:
        raise last_exc
    return ""


def generate_ai_response(prompt: str) -> str:
    try:
        if is_simple_query(prompt):
            print("[ai-router] using OpenRouter (simple query)")
            return _try_with_retries(openrouter_generate, prompt)

        try:
            print("[ai-router] using Gemini")
            return _try_with_retries(gemini_generate, prompt)
        except Exception as exc:
            print(f"[ai-router] Gemini failed, falling back to OpenRouter: {exc}")
            return _try_with_retries(openrouter_generate, prompt)

    except Exception as exc:
        print(f"[ai-router] all providers failed: {exc}")
        return "Sorry, I'm having trouble right now. Please try again shortly."
