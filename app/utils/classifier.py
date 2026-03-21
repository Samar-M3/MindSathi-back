def is_simple_query(prompt: str) -> bool:
    if not prompt:
        return True
    heavy_keywords = ["plan", "schedule", "overwhelmed", "stress", "help"]
    lowered = prompt.lower()
    if any(k in lowered for k in heavy_keywords):
        return False

    words = prompt.strip().split()
    if len(words) < 20:
        return True
    return False
