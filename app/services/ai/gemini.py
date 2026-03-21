import os
import google.generativeai as genai


def gemini_generate(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")
    resp = model.generate_content(prompt, generation_config={"max_output_tokens": 512})
    return (resp.text or "").strip()
