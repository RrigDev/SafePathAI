"""
SafePath AI — Flask backend.

Everything shown to the user comes from one of three places:

  1. crime_risk_summary.csv  (built by risk_engine.py from the real dataset)
  2. the trained ML model    (built by train_model.py, optional)
  3. OpenStreetMap services  (geocoding, nearby places, walking routes)

Nothing is invented. Where data does not exist (street-level incident
coordinates), the app says so instead of faking it.
"""

import json
import math
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from flask import Flask, jsonify, redirect, render_template, request

from city_utils import CITY_ALIASES, normalize_city

app = Flask(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

GEOCODE_URL = "https://nominatim.openstreetmap.org/search"
REVERSE_GEOCODE_URL = "https://nominatim.openstreetmap.org/reverse"
PHOTON_URL = "https://photon.komoot.io/api/"

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

HEADERS = {
    "User-Agent": (
        "SafePathAI-StudentProject/2.0 "
        "(https://github.com/scrambledgamer2806-boop/SAFEPATH-AI)"
    )
}

DEFAULT_LAT = 12.9716
DEFAULT_LNG = 77.5946

INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")

RISK_DATA_FILE = Path("crimes_indian_dataset/crime_risk_summary.csv")
MODEL_DIR = Path("model")

TIME_PERIOD_ORDER = ["Morning", "Afternoon", "Evening", "Night"]

PERIOD_HOURS = {
    "Morning": "5 AM – 12 PM",
    "Afternoon": "12 – 5 PM",
    "Evening": "5 – 9 PM",
    "Night": "9 PM – 5 AM",
}

# Same weights risk_engine.py uses — shown to the user in "Why this score".
FREQUENCY_WEIGHT = 0.60
SEVERITY_WEIGHT = 0.40

CATEGORY_FIELDS = [
    ("Violent crime", "violent_crimes"),
    ("Theft", "theft"),
    ("Robbery", "robbery"),
    ("Vehicle theft", "vehicle_theft"),
    ("Property crime", "property_crimes"),
    ("Domestic violence", "domestic_violence"),
]

LEVEL_STYLE = {
    "LOW": {
        "css": "risk-low",
        "color": "#28a745",
        "summary": (
            "Historical crime data indicates relatively lower risk "
            "during this time of day."
        ),
        "tips": [
            "Stay aware of your surroundings",
            "Keep emergency contacts easy to reach",
            "Keep valuables out of sight",
            "Let someone know where you're headed",
        ],
    },
    "MEDIUM": {
        "css": "risk-medium",
        "color": "#ffc107",
        "summary": (
            "Historical crime data indicates moderate risk during this "
            "time of day. Exercise additional caution."
        ),
        "tips": [
            "Prefer well-lit, busy roads",
            "Avoid isolated areas when possible",
            "Keep your phone charged and within reach",
            "Share your live location with someone you trust",
            "Keep valuables out of sight",
        ],
    },
    "HIGH": {
        "css": "risk-high",
        "color": "#dc3545",
        "summary": (
            "Historical crime data indicates higher risk during this "
            "time of day. Take additional precautions."
        ),
        "tips": [
            "Prefer well-lit and populated roads",
            "Avoid isolated areas when possible",
            "Share your live location with someone you trust",
            "Consider a trusted ride service instead of walking",
            "Keep your phone charged and within reach",
            "Save local emergency numbers before you set off",
        ],
    },
}


# ============================================================
# LOAD REAL RISK DATA
# ============================================================

print("Loading SafePath-AI risk data...")

ROWS = {}          # (normalised city, period) -> row dict
CITY_LABELS = {}   # normalised city -> display name
META = {"records": 0, "cities": 0}


def load_risk_data():

    try:
        df = pd.read_csv(RISK_DATA_FILE)
    except Exception as e:
        print("ERROR: could not load risk dataset:", e)
        return

    required = {"city", "time_period", "risk_score", "risk_level", "total_crimes"}

    if not required.issubset(df.columns):
        print("ERROR: risk dataset is missing columns:", required - set(df.columns))
        return

    df["_city"] = df["city"].map(normalize_city)
    df["_period"] = df["time_period"].astype(str).str.strip().str.title()
    df["risk_level"] = df["risk_level"].astype(str).str.upper()

    # incidents with no usable time ("Unknown") can't be placed in a period
    df = df[df["_period"].isin(TIME_PERIOD_ORDER)]

    # if two spellings of one city both exist, keep the larger group
    df = (
        df.sort_values("total_crimes", ascending=False)
        .drop_duplicates(["_city", "_period"])
    )

    for record in df.to_dict("records"):
        ROWS[(record["_city"], record["_period"])] = record
        CITY_LABELS.setdefault(record["_city"], str(record["city"]).strip().title())

    META["records"] = int(df["total_crimes"].sum())
    META["cities"] = len({c for c, _ in ROWS})

    print(f"Loaded {len(ROWS)} city/time risk records for {META['cities']} cities.")


load_risk_data()


def num(row, key, default=0):

    try:
        value = float(row.get(key, default))
        return default if math.isnan(value) else value
    except (TypeError, ValueError):
        return default


def known_cities():
    return {c for c, _ in ROWS}


def supported_city_names():
    return sorted(CITY_LABELS.values())


PREFERRED_CITIES = [
    "Bangalore", "Delhi", "Mumbai", "Chennai", "Kolkata",
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
]


def quick_city_names():

    """Well-known cities first, then the rest — for the homepage chips."""

    names = supported_city_names()
    first = [c for c in PREFERRED_CITIES if c in names]
    rest = [c for c in names if c not in first]

    return (first + rest)[:10]


# ============================================================
# LOAD ML MODEL (optional — the app works fine without it)
# ============================================================

ML_AVAILABLE = False
ML_METRICS = {}

try:

    import joblib

    ml_model = joblib.load(MODEL_DIR / "violent_predictor.joblib")
    city_encoder = joblib.load(MODEL_DIR / "city_encoder.joblib")
    day_encoder = joblib.load(MODEL_DIR / "day_encoder.joblib")

    ML_AVAILABLE = True

    try:
        ML_METRICS = json.loads((MODEL_DIR / "metrics.json").read_text())
    except Exception:
        ML_METRICS = {}

    print("ML model loaded.")

except Exception as e:

    print("ML model not loaded (run train_model.py to enable it):", e)


# ============================================================
# TINY TTL CACHE — keeps the free public APIs happy and the
# app fast when people search the same place twice.
# ============================================================

_cache = {}
_cache_lock = threading.Lock()


def cache_get(key):

    with _cache_lock:
        hit = _cache.get(key)

    if hit and hit[0] > time.time():
        return hit[1]

    return None


def cache_put(key, value, ttl):

    with _cache_lock:

        if len(_cache) > 500:
            _cache.clear()

        _cache[key] = (time.time() + ttl, value)


# ============================================================
# CITY DETECTION
# ============================================================

# Approximate city centres. Used ONLY as a last resort when every online
# geocoder is unreachable, so a plain city name still works.
CITY_CENTERS = {
    "agra": (27.1767, 78.0081), "ahmedabad": (23.0225, 72.5714),
    "bangalore": (12.9716, 77.5946), "bhopal": (23.2599, 77.4126),
    "chennai": (13.0827, 80.2707), "delhi": (28.6139, 77.2090),
    "faridabad": (28.4089, 77.3178), "ghaziabad": (28.6692, 77.4538),
    "hyderabad": (17.3850, 78.4867), "indore": (22.7196, 75.8577),
    "jaipur": (26.9124, 75.7873), "kalyan": (19.2437, 73.1355),
    "kanpur": (26.4499, 80.3319), "kolkata": (22.5726, 88.3639),
    "lucknow": (26.8467, 80.9462), "ludhiana": (30.9010, 75.8573),
    "meerut": (28.9845, 77.7064), "mumbai": (19.0760, 72.8777),
    "nagpur": (21.1458, 79.0882), "nashik": (19.9975, 73.7898),
    "patna": (25.5941, 85.1376), "pune": (18.5204, 73.8567),
    "rajkot": (22.3039, 70.8022), "srinagar": (34.0837, 74.7973),
    "surat": (21.1702, 72.8311), "thane": (19.2183, 72.9781),
    "varanasi": (25.3176, 82.9739), "vasai": (19.4259, 72.8225),
    "visakhapatnam": (17.6868, 83.2185),
}


def find_known_city(*texts):

    """Find a dataset city (or alias such as Bengaluru) as a whole word."""

    cities = known_cities()

    if not cities:
        return ""

    haystack = " | ".join(str(t).lower() for t in texts if t)

    candidates = {c: c for c in cities}
    candidates.update({
        alias: target
        for alias, target in CITY_ALIASES.items()
        if target in cities
    })

    for name in sorted(candidates, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", haystack):
            return candidates[name]

    return ""


def city_from_address(address, display_name="", typed=""):

    """
    Work out which dataset city a geocoder result belongs to. Cities can
    hide under many address keys (Delhi often has no "city" at all), so we
    try them all, then fall back to matching the text.
    """

    cities = known_cities()

    for key in (
        "city", "town", "municipality", "city_district",
        "village", "county", "state_district", "state",
    ):
        value = normalize_city(address.get(key, ""))

        if value and (not cities or value in cities):
            return value

    found = find_known_city(display_name, typed)

    if found:
        return found

    # nothing matched our dataset — return a best guess so the error
    # message can name the place that has no data
    for key in ("city", "town", "municipality", "village", "county"):
        value = normalize_city(address.get(key, ""))

        if value:
            return value

    return ""


# ============================================================
# GEOCODING
# ============================================================

def _nominatim_search(location_str):

    response = requests.get(
        GEOCODE_URL,
        params={
            "q": location_str,
            "format": "json",
            "limit": 1,
            "addressdetails": 1,
            "countrycodes": "in",
        },
        headers=HEADERS,
        timeout=8,
    )

    response.raise_for_status()

    results = response.json()

    if not results:
        return None

    result = results[0]

    return (
        float(result["lat"]),
        float(result["lon"]),
        result.get("display_name", location_str),
        result.get("address", {}),
    )


def _photon_search(location_str):

    """Backup geocoder, used if Nominatim blocks or rate-limits the server."""

    response = requests.get(
        PHOTON_URL,
        params={"q": location_str, "limit": 1},
        headers=HEADERS,
        timeout=8,
    )

    response.raise_for_status()

    features = response.json().get("features", [])

    if not features:
        return None

    feature = features[0]
    props = feature.get("properties", {})
    lng, lat = feature["geometry"]["coordinates"]

    address = {
        "city": props.get("city") or props.get("name", ""),
        "county": props.get("county", ""),
        "state": props.get("state", ""),
    }

    display = ", ".join(
        part for part in (props.get("name"), props.get("city"), props.get("state"))
        if part
    ) or location_str

    return float(lat), float(lng), display, address


def geocode_location(location_str):

    """Returns: lat, lng, display_name, city, found"""

    key = ("geo", location_str.strip().lower())
    cached = cache_get(key)

    if cached:
        return cached

    for search in (_nominatim_search, _photon_search):

        try:
            hit = search(location_str)

            if hit:
                lat, lng, display_name, address = hit
                city = city_from_address(address, display_name, location_str)
                result = (lat, lng, display_name, city, True)
                cache_put(key, result, 6 * 3600)
                return result

        except Exception as e:
            print(f"Geocoding error ({search.__name__}):", e)

    # last resort: a plain city name we have data for
    typed_city = find_known_city(location_str)

    if typed_city and typed_city in CITY_CENTERS:
        lat, lng = CITY_CENTERS[typed_city]
        print(f"Geocoders unreachable — using built-in centre for {typed_city}")
        return lat, lng, f"{typed_city.title()} (city centre)", typed_city, True

    return DEFAULT_LAT, DEFAULT_LNG, location_str, "", False


def reverse_geocode_city(lat, lng):

    try:
        response = requests.get(
            REVERSE_GEOCODE_URL,
            params={
                "lat": lat,
                "lon": lng,
                "format": "json",
                "zoom": 10,
                "addressdetails": 1,
            },
            headers=HEADERS,
            timeout=8,
        )

        response.raise_for_status()
        result = response.json()

        return city_from_address(
            result.get("address", {}),
            result.get("display_name", ""),
        )

    except Exception as e:
        print("Reverse geocoding error:", e)
        return ""


def parse_gps_input(location_str):

    """Parse raw 'lat,lng' input. Returns (lat, lng) or None."""

    parts = location_str.split(",")

    if len(parts) != 2:
        return None

    try:
        lat = float(parts[0].strip())
        lng = float(parts[1].strip())
    except ValueError:
        return None

    if -90 <= lat <= 90 and -180 <= lng <= 180:
        return lat, lng

    return None


# ============================================================
# TIME
# ============================================================

def get_current_time_period():

    now = datetime.now(INDIA_TIMEZONE)
    hour = now.hour

    if 5 <= hour < 12:
        period = "Morning"
    elif 12 <= hour < 17:
        period = "Afternoon"
    elif 17 <= hour < 21:
        period = "Evening"
    else:
        period = "Night"

    return now, period


def fmt_hour(hour):

    h12 = hour % 12 or 12

    return f"{h12} {'AM' if hour < 12 else 'PM'}"


def ordinal(n):

    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

    return f"{n}{suffix}"


# ============================================================
# RISK DATA VIEWS
# ============================================================

def period_view(row):

    level = str(row["risk_level"]).upper()
    style = LEVEL_STYLE.get(level, LEVEL_STYLE["HIGH"])

    risk_score = round(num(row, "risk_score"), 1)

    return {
        "period": row["_period"],
        "hours": PERIOD_HOURS.get(row["_period"], ""),
        "score": round(max(0, min(100, 100 - risk_score))),
        "risk_score": risk_score,
        "risk_level": level.title(),
        "risk_class": style["css"],
        "map_color": style["color"],
        "summary": style["summary"],
        "total_crimes": int(num(row, "total_crimes")),
    }


def get_crime_data(city, time_period):

    row = ROWS.get((normalize_city(city), time_period))

    if row is None:
        return None

    view = period_view(row)
    style = LEVEL_STYLE.get(str(row["risk_level"]).upper(), LEVEL_STYLE["HIGH"])

    view["tips"] = style["tips"]

    view["stats"] = {
        "total_crimes": int(num(row, "total_crimes")),
        "violent_crimes": int(num(row, "violent_crimes")),
        "theft": int(num(row, "theft")),
        "robbery": int(num(row, "robbery")),
        "vehicle": int(num(row, "vehicle_theft")),
        "property_crimes": int(num(row, "property_crimes")),
        "domestic_violence": int(num(row, "domestic_violence")),
    }

    return view


def get_all_periods(city):

    city = normalize_city(city)

    return {
        period: period_view(ROWS[(city, period)])
        for period in TIME_PERIOD_ORDER
        if (city, period) in ROWS
    }


def get_score_components(city, period):

    """
    The exact recipe risk_engine.py uses:
        risk = 0.60 x frequency score + 0.40 x severity score
    Returned so the page can show *why* the number is what it is.
    """

    row = ROWS.get((normalize_city(city), period))

    if row is None:
        return None

    try:
        freq = float(row["frequency_score"])
        sev = float(row["severity_score"])
        ratio = float(row["frequency_ratio"])
        per_crime = float(row["severity_per_crime"])
    except (KeyError, TypeError, ValueError):
        return None

    if any(math.isnan(x) for x in (freq, sev, ratio, per_crime)):
        return None

    freq_part = freq * FREQUENCY_WEIGHT
    sev_part = sev * SEVERITY_WEIGHT
    total = max(freq_part + sev_part, 0.1)

    return {
        "frequency_score": round(freq, 1),
        "severity_score": round(sev, 1),
        "frequency_points": round(freq_part, 1),
        "severity_points": round(sev_part, 1),
        "frequency_share": round(freq_part / total * 100),
        "severity_share": round(sev_part / total * 100),
        "ratio": round(ratio, 2),
        "severity_per_crime": round(per_crime, 1),
    }


def get_crime_profile(city, period):

    """
    Share of this period's incidents in each category, next to the share
    across the whole day for the same city — so the page can show what
    makes this time of day different.
    """

    city = normalize_city(city)
    now_row = ROWS.get((city, period))

    if now_row is None:
        return []

    rows = [ROWS[(city, p)] for p in TIME_PERIOD_ORDER if (city, p) in ROWS]

    total_now = num(now_row, "total_crimes")
    total_all = sum(num(r, "total_crimes") for r in rows)

    if not total_now:
        return []

    profile = []

    for label, field in CATEGORY_FIELDS:

        count = int(num(now_row, field))
        share_now = count / total_now * 100

        if len(rows) >= 2 and total_all:
            share_all = sum(num(r, field) for r in rows) / total_all * 100
            diff = round(share_now - share_all, 1)
            share_all = round(share_all, 1)
        else:
            share_all, diff = None, None

        profile.append({
            "label": label,
            "count": count,
            "pct": round(share_now, 1),
            "share_all": share_all,
            "diff": diff,
        })

    profile.sort(key=lambda item: item["count"], reverse=True)

    return profile


def get_best_time_rows(periods, current_period):

    rows = [
        {
            "period": v["period"],
            "hours": v["hours"],
            "risk_score": v["risk_score"],
            "risk_level": v["risk_level"],
            "map_color": v["map_color"],
            "total_crimes": v["total_crimes"],
            "is_current": v["period"] == current_period,
        }
        for v in periods.values()
    ]

    if not rows:
        return [], None, None

    best = min(rows, key=lambda r: r["risk_score"])
    worst = max(rows, key=lambda r: r["risk_score"])

    return rows, best, worst


def build_assessment(city, current, periods, best, worst):

    """Plain-English summary, written only from numbers we actually have."""

    if not periods:
        return ""

    ranked = sorted(periods.values(), key=lambda v: v["risk_score"])
    position = next(
        (i for i, v in enumerate(ranked, start=1) if v["period"] == current["period"]),
        None,
    )

    text = (
        f"{city.title()} shows {current['risk_level'].lower()} historical risk "
        f"during the {current['period'].lower()} period "
        f"(safety score {current['score']}/100)."
    )

    if len(ranked) >= 2 and position:

        if position == 1:
            text += f" It is the historically safest part of the day in {city.title()}."
        elif position == len(ranked):
            text += (
                f" It is the historically riskiest part of the day — "
                f"{best['period']} is safer ({best['risk_score']}/100)."
            )
        else:
            text += (
                f" It ranks {ordinal(position)} safest of {len(ranked)} periods; "
                f"{best['period']} is the safest ({best['risk_score']}/100)."
            )

    return text


# ============================================================
# MACHINE LEARNING
# ============================================================

def get_ml_insight(city, now):

    """
    Hour-by-hour estimate of the share of incidents that are violent for
    this city / weekday / month, from the trained Random Forest.
    """

    if not ML_AVAILABLE:
        return None

    try:
        city = normalize_city(city)
        day = now.strftime("%A")

        if city not in city_encoder.classes_ or day not in day_encoder.classes_:
            return None

        classes = list(ml_model.classes_)

        if 1 not in classes:
            return None

        features = pd.DataFrame({
            "city_enc": [int(city_encoder.transform([city])[0])] * 24,
            "hour": list(range(24)),
            "day_enc": [int(day_encoder.transform([day])[0])] * 24,
            "month": [now.month] * 24,
        })[["city_enc", "hour", "day_enc", "month"]]

        probs = ml_model.predict_proba(features)[:, classes.index(1)] * 100
        hourly = [round(float(p), 1) for p in probs]

        peak = max(range(24), key=lambda h: hourly[h])
        calm = min(range(24), key=lambda h: hourly[h])
        spread = round(hourly[peak] - hourly[calm], 1)

        importances = ML_METRICS.get("feature_importances") or {}

        top_feature = (
            max(importances, key=importances.get) if importances else None
        )

        return {
            "now_pct": hourly[now.hour],
            "hourly": hourly,
            "peak_hour": fmt_hour(peak),
            "peak_pct": hourly[peak],
            "calm_hour": fmt_hour(calm),
            "calm_pct": hourly[calm],
            "spread": spread,
            "weak_signal": spread < 3,
            "weekday": day,
            "top_feature": top_feature,
            "importances": sorted(
                ({"label": k, "pct": round(v * 100, 1)} for k, v in importances.items()),
                key=lambda item: -item["pct"],
            ),
            "metrics": ML_METRICS,
        }

    except Exception as e:
        print("ML insight error:", e)
        return None


# ============================================================
# NEARBY PLACES (OpenStreetMap Overpass API)
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):

    r = 6371.0

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )

    return 2 * r * math.asin(math.sqrt(a))


def overpass_query(query):

    """Returns a list of elements, or None if every server failed."""

    for url in OVERPASS_URLS:
        try:
            response = requests.post(
                url, data={"data": query}, headers=HEADERS, timeout=9
            )
            response.raise_for_status()
            return response.json().get("elements", [])

        except Exception as e:
            print(f"Overpass error ({url}):", e)

    return None


def find_nearest(lat, lng, tag, radii, label):

    """
    Nearest OSM feature with the given tag. Searches small circles first and
    widens only if nothing is found, so the nearest one is never missed just
    because a large circle returned too many results.
    """

    for radius in radii:

        query = (
            "[out:json][timeout:8];"
            f"nwr[{tag}](around:{radius},{lat},{lng});"
            "out center 200;"
        )

        elements = overpass_query(query)

        if elements is None:
            return None  # service unreachable — don't keep retrying

        best, best_dist = None, None

        for el in elements:

            if "lat" in el and "lon" in el:
                elat, elon = el["lat"], el["lon"]
            elif "center" in el:
                elat, elon = el["center"]["lat"], el["center"]["lon"]
            else:
                continue

            dist = haversine_km(lat, lng, elat, elon)

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = {
                    "lat": elat,
                    "lng": elon,
                    "name": el.get("tags", {}).get("name") or label,
                    "distance_km": round(dist, 2),
                }

        if best:
            return best

    return None


def get_destinations(lat, lng):

    key = ("dest", round(lat, 3), round(lng, 3))
    cached = cache_get(key)

    if cached:
        return cached

    def find_transit():
        return (
            find_nearest(lat, lng, '"highway"="bus_stop"', [500, 1200, 3000], "Bus stop")
            or find_nearest(lat, lng, '"railway"="station"', [3000, 8000], "Station")
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_hospital = pool.submit(
            find_nearest, lat, lng, '"amenity"="hospital"', [1500, 4000, 10000], "Hospital"
        )
        f_police = pool.submit(
            find_nearest, lat, lng, '"amenity"="police"', [1500, 4000, 10000], "Police station"
        )
        f_transit = pool.submit(find_transit)

        found = {
            "hospital": f_hospital.result(),
            "police": f_police.result(),
            "transit": f_transit.result(),
        }

    fallback = {
        "hospital": (-0.022, 0.015, "Hospital (approximate — live lookup unavailable)"),
        "police": (0.009, 0.021, "Police station (approximate — live lookup unavailable)"),
        "transit": (0.014, -0.010, "Transit stop (approximate — live lookup unavailable)"),
    }

    emoji = {"hospital": "🏥", "police": "🚔", "transit": "🚌"}

    result = {}

    for kind in ("hospital", "police", "transit"):

        real = found[kind]

        if real:
            result[kind] = {
                "lat": real["lat"],
                "lng": real["lng"],
                "label": real["name"],
                "distance_km": real["distance_km"],
                "found": True,
                "emoji": emoji[kind],
            }
        else:
            dlat, dlng, note = fallback[kind]
            result[kind] = {
                "lat": lat + dlat,
                "lng": lng + dlng,
                "label": note,
                "distance_km": None,
                "found": False,
                "emoji": emoji[kind],
            }

    if any(item["found"] for item in result.values()):
        cache_put(key, result, 3600)

    return result


# ============================================================
# PAGES
# ============================================================

def render_home(error=None, query=""):

    return render_template(
        "index.html",
        error=error,
        query=query,
        cities=supported_city_names(),
        quick_cities=quick_city_names(),
        meta=META,
    )


@app.route("/")
def home():
    return render_home()


@app.route("/analyze", methods=["GET", "POST"])
def analyze():

    raw_input = (request.values.get("location") or "").strip()[:200]

    if not raw_input:

        if request.method == "GET":
            return redirect("/")

        return render_home(error="Please enter a location.")

    if not ROWS:
        return render_home(
            error=(
                "The crime dataset isn't loaded on the server yet. "
                "Check /health to see what's missing."
            ),
            query=raw_input,
        )

    gps = parse_gps_input(raw_input)

    if gps:
        lat, lng = gps
        display_location = f"Your location ({lat:.4f}, {lng:.4f})"
        geocoded = True
        city = reverse_geocode_city(lat, lng)
    else:
        lat, lng, display_location, city, geocoded = geocode_location(raw_input)

    if not geocoded:
        return render_home(
            error=(
                "Couldn't find that location. Try a more specific place name, "
                "or add the city — e.g. “Koramangala, Bangalore”."
            ),
            query=raw_input,
        )

    if not city:
        city = find_known_city(raw_input)

    if not city:
        return render_home(
            error=(
                "We found the place but couldn't tell which city it's in. "
                "Try adding the city name — e.g. “Koramangala, Bangalore”."
            ),
            query=raw_input,
        )

    now, time_period = get_current_time_period()

    data = get_crime_data(city, time_period)

    if data is None:
        covered = ", ".join(supported_city_names())
        return render_home(
            error=(
                f"We don't have historical crime data for {city.title()} yet. "
                f"Cities covered: {covered}."
            ),
            query=raw_input,
        )

    periods = get_all_periods(city)
    best_time_rows, best_time, worst_time = get_best_time_rows(periods, time_period)

    return render_template(
        "result.html",
        query=raw_input,
        location=display_location,
        lat=lat,
        lng=lng,
        city=city.title(),
        time_period=time_period,
        period_hours=PERIOD_HOURS[time_period],
        current_time=now.strftime("%I:%M %p").lstrip("0"),
        current_hour=now.hour,
        data=data,
        periods=periods,
        best_time_rows=best_time_rows,
        best_time=best_time,
        worst_time=worst_time,
        assessment=build_assessment(city, data, periods, best_time, worst_time),
        components=get_score_components(city, time_period),
        profile=get_crime_profile(city, time_period),
        ml=get_ml_insight(city, now),
        destinations=get_destinations(lat, lng),
        meta=META,
        ml_metrics=ML_METRICS,
        ml_available=ML_AVAILABLE,
    )


@app.route("/health")
def health():

    """Open /health on your live site to confirm data + model loaded."""

    return jsonify(
        status="ok" if ROWS else "no-data",
        risk_rows=len(ROWS),
        cities=META["cities"],
        incidents_covered=META["records"],
        ml_model_loaded=ML_AVAILABLE,
        ml_metrics_loaded=bool(ML_METRICS),
    )


@app.errorhandler(404)
def not_found(_):

    return render_template(
        "error.html",
        code=404,
        title="Page not found",
        message="That page doesn't exist. Let's get you back to the search.",
    ), 404


@app.errorhandler(500)
def server_error(_):

    traceback.print_exc()

    return render_template(
        "error.html",
        code=500,
        title="Something went wrong",
        message=(
            "The server hit an unexpected problem while analysing that "
            "location. Please try again."
        ),
    ), 500


if __name__ == "__main__":

    app.run(debug=True)
