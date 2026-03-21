import re

KEYWORDS = [
    "latest",
    "news",
    "today",
    "current",
    "price",
    "who won",
    "update",
    "recent",
]

def needs_web_search(query: str) -> bool:
    lower_q = query.lower()
    return any(k in lower_q for k in KEYWORDS)
