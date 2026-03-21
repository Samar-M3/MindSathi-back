import os
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS = [
    "mistralai/mistral-7b-instruct",
    "meta-llama/llama-3-8b-instruct",
]


def _make_request(prompt: str, model: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.7,
    }
    resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def openrouter_generate(prompt: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    for model in OPENROUTER_MODELS:
        try:
            return _make_request(prompt, model, api_key)
        except Exception as exc:
            print(f"[openrouter] model {model} failed: {exc}")
            continue
    return "Sorry, I ran into a hiccup. Please try again in a moment."
