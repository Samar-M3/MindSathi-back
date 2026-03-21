import os
from typing import List, Dict
import httpx

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_URL = "https://google.serper.dev/search"


async def web_search(query: str, num_results: int = 5) -> List[Dict]:
    if not SERPER_API_KEY:
        raise ValueError("SERPER_API_KEY not set")

    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results}

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(SERPER_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    items = data.get("organic", [])[:num_results]
    results = []
    for item in items:
        results.append(
            {
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "link": item.get("link"),
            }
        )
    return results


def build_context(results: List[Dict]) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r.get('title','')}: {r.get('snippet','')} (source: {r.get('link','')})")
    return "\n".join(lines)
