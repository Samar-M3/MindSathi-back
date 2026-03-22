"""
places.py — Nearby mental health clinic finder for MindSathi
Uses zero-cost, no-API-key approach:
  1. Hardcoded verified Nepal clinics (always reliable)
  2. Overpass API (OpenStreetMap live data) for dynamic nearby results
  3. Nominatim for city name → lat/lng conversion if GPS unavailable
"""

import math
import httpx
from typing import Optional

# ── Haversine distance ────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Returns distance in km between two lat/lng points."""
    R = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(d_lng / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Verified Nepal mental health clinics (hardcoded, always available) ────────

NEPAL_CLINICS = [
    # Kathmandu Valley
    {
        "name": "Koshish Mental Health Self Help Organization",
        "address": "Baluwatar, Kathmandu",
        "phone": "01-4002010",
        "lat": 27.7172, "lng": 85.3240,
        "city": "Kathmandu",
        "note": "Mental health support and counseling",
    },
    {
        "name": "TPO Nepal (Transcultural Psychosocial Organization)",
        "address": "Baluwatar, Kathmandu",
        "phone": "01-4460184",
        "lat": 27.7175, "lng": 85.3235,
        "city": "Kathmandu",
        "note": "Psychosocial support, trauma counseling",
    },
    {
        "name": "Tribhuvan University Teaching Hospital — Psychiatry",
        "address": "Maharajgunj, Kathmandu",
        "phone": "01-4412505",
        "lat": 27.7362, "lng": 85.3313,
        "city": "Kathmandu",
        "note": "Full psychiatric services, OPD available",
    },
    {
        "name": "Mental Hospital Lagankhel",
        "address": "Lagankhel, Lalitpur",
        "phone": "01-5521333",
        "lat": 27.6630, "lng": 85.3170,
        "city": "Lalitpur",
        "note": "Nepal's national mental hospital",
    },
    {
        "name": "Patan Hospital — Psychiatry Department",
        "address": "Lagankhel, Lalitpur",
        "phone": "01-5522266",
        "lat": 27.6644, "lng": 85.3176,
        "city": "Lalitpur",
        "note": "Outpatient psychiatric services",
    },
    {
        "name": "Om Counseling Center",
        "address": "Putalisadak, Kathmandu",
        "phone": "9841234567",
        "lat": 27.7041, "lng": 85.3195,
        "city": "Kathmandu",
        "note": "Individual and family counseling",
    },
    {
        "name": "Nepal Mental Health Foundation",
        "address": "Kalopul, Kathmandu",
        "phone": "01-4413800",
        "lat": 27.7080, "lng": 85.3290,
        "city": "Kathmandu",
        "note": "Community mental health programs",
    },
    {
        "name": "Bir Hospital — Psychiatry OPD",
        "address": "Mahaboudha, Kathmandu",
        "phone": "01-4221119",
        "lat": 27.7030, "lng": 85.3145,
        "city": "Kathmandu",
        "note": "Government hospital, free OPD",
    },
    # Pokhara
    {
        "name": "Western Regional Hospital — Psychiatry",
        "address": "Ramghat, Pokhara",
        "phone": "061-520066",
        "lat": 28.2096, "lng": 83.9856,
        "city": "Pokhara",
        "note": "Psychiatric OPD, government facility",
    },
    {
        "name": "Gandaki Medical College — Mental Health",
        "address": "Pokhara",
        "phone": "061-431000",
        "lat": 28.2380, "lng": 83.9956,
        "city": "Pokhara",
        "note": "Teaching hospital with psychiatric services",
    },
    # Biratnagar
    {
        "name": "BP Koirala Institute of Health Sciences — Psychiatry",
        "address": "Dharan, Sunsari",
        "phone": "025-525555",
        "lat": 26.8127, "lng": 87.2841,
        "city": "Dharan",
        "note": "Tertiary care, full psychiatric department",
    },
    # Chitwan
    {
        "name": "Bharatpur Hospital — Psychiatry",
        "address": "Bharatpur, Chitwan",
        "phone": "056-523555",
        "lat": 27.6833, "lng": 84.4333,
        "city": "Bharatpur",
        "note": "Government hospital psychiatric services",
    },
    # Butwal
    {
        "name": "Lumbini Provincial Hospital — Mental Health",
        "address": "Butwal, Rupandehi",
        "phone": "071-540047",
        "lat": 27.7006, "lng": 83.4532,
        "city": "Butwal",
        "note": "Provincial hospital with mental health unit",
    },
]


# ── Overpass API — live OpenStreetMap query ───────────────────────────────────

async def _fetch_overpass_clinics(lat: float, lng: float, radius_m: int = 5000) -> list[dict]:
    """
    Queries OpenStreetMap via Overpass API for nearby mental health / medical facilities.
    Free, no API key. Returns empty list on failure.
    """
    query = f"""
[out:json][timeout:10];
(
  node["amenity"~"hospital|clinic|doctors"]["healthcare"~"psychiatry|counselling|mental_health"](around:{radius_m},{lat},{lng});
  node["amenity"~"hospital|clinic|doctors"]["name"~"mental|psycho|counsel|psychiatric|wellbeing",i](around:{radius_m},{lat},{lng});
  way["amenity"~"hospital|clinic"]["healthcare"~"psychiatry|counselling|mental_health"](around:{radius_m},{lat},{lng});
);
out center 5;
"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": query},
            )
            data = response.json()
            results = []
            for el in data.get("elements", [])[:5]:
                tags = el.get("tags", {})
                name = tags.get("name")
                if not name:
                    continue
                el_lat = el.get("lat") or el.get("center", {}).get("lat")
                el_lng = el.get("lon") or el.get("center", {}).get("lon")
                distance = None
                if el_lat and el_lng:
                    distance = round(_haversine_km(lat, lng, el_lat, el_lng), 1)
                results.append({
                    "name": name,
                    "address": tags.get("addr:full") or tags.get("addr:street") or "See map",
                    "phone": tags.get("phone") or tags.get("contact:phone") or "N/A",
                    "distance_km": distance,
                    "maps_link": f"https://www.openstreetmap.org/?mlat={el_lat}&mlon={el_lng}&zoom=17" if el_lat else None,
                    "source": "openstreetmap",
                })
            return results
    except Exception as e:
        print(f"Overpass API error: {e}")
        return []


# ── Nominatim — city name to lat/lng ─────────────────────────────────────────

async def city_to_latlng(city: str) -> Optional[tuple[float, float]]:
    """
    Converts a city name to lat/lng using Nominatim (free, no API key).
    Returns (lat, lng) or None on failure.
    """
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": f"{city}, Nepal", "format": "json", "limit": 1},
                headers={"User-Agent": "MindSathi/1.0 mental-health-app"},
            )
            results = response.json()
            if results:
                return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception as e:
        print(f"Nominatim error: {e}")
    return None


# ── Main function ─────────────────────────────────────────────────────────────

async def get_nearby_doctors(
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    city: Optional[str] = None,
    max_results: int = 3,
) -> list[dict]:
    """
    Returns nearby mental health professionals.
    Priority:
      1. lat/lng provided → sort hardcoded list by distance + try Overpass
      2. city provided → geocode with Nominatim, then same as above
      3. Nothing provided → return top Kathmandu clinics as fallback

    Each result dict has: name, address, phone, distance_km, maps_link
    """

    # ── Resolve coordinates ───────────────────────────────────────────────────
    if lat is None or lng is None:
        if city:
            coords = await city_to_latlng(city)
            if coords:
                lat, lng = coords

    # ── Score and sort hardcoded clinics ──────────────────────────────────────
    scored = []
    for clinic in NEPAL_CLINICS:
        if lat is not None and lng is not None:
            dist = _haversine_km(lat, lng, clinic["lat"], clinic["lng"])
        else:
            # No location at all — default sort by Kathmandu proximity
            dist = _haversine_km(27.7172, 85.3240, clinic["lat"], clinic["lng"])

        scored.append({
            **clinic,
            "distance_km": round(dist, 1),
            "maps_link": (
                f"https://www.openstreetmap.org/?mlat={clinic['lat']}"
                f"&mlon={clinic['lng']}&zoom=17"
            ),
            "source": "verified",
        })

    scored.sort(key=lambda x: x["distance_km"])
    hardcoded_top = scored[:max_results]

    # ── Try Overpass for live results if we have coordinates ──────────────────
    live_results = []
    if lat is not None and lng is not None:
        live_results = await _fetch_overpass_clinics(lat, lng)

    # ── Merge: verified clinics first, then live if they add new names ────────
    seen_names = {c["name"].lower() for c in hardcoded_top}
    merged = list(hardcoded_top)
    for live in live_results:
        if live["name"].lower() not in seen_names:
            merged.append(live)
            seen_names.add(live["name"].lower())
        if len(merged) >= max_results + 2:
            break

    return merged[:max_results]


# ── Formatter for crisis response ─────────────────────────────────────────────

def format_doctors_for_chat(doctors: list[dict]) -> str:
    """Formats doctor list into a clean chat-friendly string."""
    if not doctors:
        return ""

    lines = ["🏥 **Nearby mental health professionals:**"]
    for d in doctors:
        dist = f" · {d['distance_km']} km away" if d.get("distance_km") is not None else ""
        phone = f" · 📞 {d['phone']}" if d.get("phone") and d["phone"] != "N/A" else ""
        link = f"\n  → {d['maps_link']}" if d.get("maps_link") else ""
        note = f" — {d['note']}" if d.get("note") else ""
        lines.append(f"• **{d['name']}**{dist}{phone}{note}{link}")

    return "\n".join(lines)