from flask import Flask, render_template, request
import requests
import pandas as pd
import math
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from city_utils import normalize_city

app = Flask(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

GEOCODE_URL = "https://nominatim.openstreetmap.org/search"
REVERSE_GEOCODE_URL = "https://nominatim.openstreetmap.org/reverse"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

HEADERS = {
    "User-Agent": "SafePathAI-StudentProject/1.0"
}

# Default fallback center
DEFAULT_LAT = 12.9716
DEFAULT_LNG = 77.5946

# India timezone
INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")

# Real risk dataset
RISK_DATA_FILE = Path(
    "crimes_indian_dataset/crime_risk_summary.csv"
)

TIME_PERIOD_ORDER = ["Morning", "Afternoon", "Evening", "Night"]


# ============================================================
# LOAD REAL RISK DATA
# ============================================================

print("Loading SafePath-AI risk data...")

try:

    risk_df = pd.read_csv(RISK_DATA_FILE)

    print(
        f"Loaded {len(risk_df):,} city/time risk records."
    )

except Exception as e:

    print("ERROR: Could not load risk dataset.")
    print(e)

    risk_df = pd.DataFrame()


# ============================================================
# LOAD ML MODEL (optional — app still works without it)
# ============================================================

ML_AVAILABLE = False

try:

    import joblib

    MODEL_DIR = Path("model")

    ml_model = joblib.load(MODEL_DIR / "violent_predictor.joblib")
    city_encoder = joblib.load(MODEL_DIR / "city_encoder.joblib")
    day_encoder = joblib.load(MODEL_DIR / "day_encoder.joblib")

    ML_AVAILABLE = True

    print("ML model loaded — ML Insight panel enabled.")

except Exception as e:

    print(
        "ML model not found or failed to load "
        "(run train_model.py to enable it). Continuing without it."
    )
    print(e)


# ============================================================
# GEOCODING
# ============================================================

def geocode_location(location_str):

    """
    Convert a typed address/place into real coordinates
    using OpenStreetMap Nominatim.

    Returns:

        lat
        lng
        display_name
        city
        found
    """

    try:

        response = requests.get(

            GEOCODE_URL,

            params={
                "q": location_str,
                "format": "json",
                "limit": 1,
                "addressdetails": 1,
            },

            headers=HEADERS,

            timeout=8,
        )

        response.raise_for_status()

        results = response.json()

        if results:

            result = results[0]

            lat = float(result["lat"])
            lng = float(result["lon"])

            display_name = result.get(
                "display_name",
                location_str
            )

            address = result.get(
                "address",
                {}
            )

            city = (
                address.get("city")
                or address.get("town")
                or address.get("municipality")
                or address.get("village")
                or ""
            )

            return (
                lat,
                lng,
                display_name,
                normalize_city(city),
                True
            )

    except Exception as e:

        print("Geocoding error:", e)

    return (
        DEFAULT_LAT,
        DEFAULT_LNG,
        location_str,
        "",
        False
    )


# ============================================================
# REVERSE GEOCODING
# ============================================================

def reverse_geocode_city(lat, lng):

    """Convert GPS coordinates into a city name."""

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

        address = result.get(
            "address",
            {}
        )

        city = (
            address.get("city")
            or address.get("town")
            or address.get("municipality")
            or address.get("village")
            or ""
        )

        return normalize_city(city)

    except Exception as e:

        print("Reverse geocoding error:", e)

        return ""


# ============================================================
# GPS INPUT
# ============================================================

def parse_gps_input(location_str):

    """Parse raw "lat,lng" input. Returns (lat, lng) or None."""

    if "," not in location_str:

        return None

    parts = location_str.split(",")

    if len(parts) != 2:

        return None

    try:

        lat = float(parts[0].strip())
        lng = float(parts[1].strip())

        if -90 <= lat <= 90 and -180 <= lng <= 180:

            return lat, lng

    except ValueError:

        return None

    return None


# ============================================================
# CURRENT TIME PERIOD
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


# ============================================================
# REAL CRIME RISK LOOKUP
# ============================================================

def get_crime_data(city, time_period):

    """
    Look up real historical crime statistics from
    crime_risk_summary.csv.

    IMPORTANT: this is city + time based historical risk.
    It is NOT street-level crime prediction because the
    source dataset does not contain crime coordinates.
    """

    if risk_df.empty:

        return None

    city = normalize_city(city)

    matches = risk_df[
        (risk_df["city"].astype(str).str.strip().str.lower() == city)
        &
        (risk_df["time_period"].astype(str).str.strip().str.lower() == time_period.lower())
    ]

    if matches.empty:

        return None

    row = matches.iloc[0]

    risk_score = float(row["risk_score"])
    risk_level = str(row["risk_level"]).upper()

    safety_score = round(max(0, min(100, 100 - risk_score)))

    if risk_level == "LOW":

        risk_class = "risk-low"
        map_color = "#28a745"

        summary = (
            "Historical crime data indicates "
            "relatively lower risk during this time."
        )

        tips = [
            "Stay aware of your surroundings",
            "Keep emergency contacts accessible",
            "Keep valuables secure",
            "Continue using normal safety precautions",
        ]

    elif risk_level == "MEDIUM":

        risk_class = "risk-medium"
        map_color = "#ffc107"

        summary = (
            "Historical crime data indicates "
            "moderate risk during this time. "
            "Exercise additional caution."
        )

        tips = [
            "Prefer well-lit roads",
            "Avoid isolated areas when possible",
            "Keep your phone accessible",
            "Consider travelling with others at night",
        ]

    else:

        risk_class = "risk-high"
        map_color = "#dc3545"

        summary = (
            "Historical crime data indicates "
            "higher risk during this time. "
            "Take additional precautions."
        )

        tips = [
            "Prefer well-lit and populated roads",
            "Avoid isolated areas when possible",
            "Share your trip with someone you trust",
            "Consider using a trusted transport service",
        ]

    stats = {
        "total_crimes": int(row["total_crimes"]),
        "violent_crimes": int(row["violent_crimes"]),
        "theft": int(row["theft"]),
        "robbery": int(row["robbery"]),
        "vehicle": int(row["vehicle_theft"]),
        "property_crimes": int(row["property_crimes"]),
        "domestic_violence": int(row["domestic_violence"]),
    }

    return {
        "score": safety_score,
        "risk_score": round(risk_score, 1),
        "risk_level": risk_level.title(),
        "risk_class": risk_class,
        "map_color": map_color,
        "summary": summary,
        "tips": tips,
        "city": str(row["city"]),
        "time_period": str(row["time_period"]),
        "stats": stats,
    }


# ============================================================
# WHY IS THIS AREA RISKY? — breakdown of the stats we already
# have, turned into a readable, ranked explanation.
# ============================================================

def get_risk_breakdown(stats):

    total = stats["total_crimes"] or 1

    items = [
        ("Violent crime", stats["violent_crimes"]),
        ("Theft", stats["theft"]),
        ("Robbery", stats["robbery"]),
        ("Vehicle theft", stats["vehicle"]),
        ("Property crime", stats["property_crimes"]),
        ("Domestic violence", stats["domestic_violence"]),
    ]

    breakdown = [
        {
            "label": label,
            "count": count,
            "pct": round((count / total) * 100, 1),
        }
        for label, count in items
    ]

    breakdown.sort(key=lambda x: x["count"], reverse=True)

    return breakdown


# ============================================================
# BEST TIME TO TRAVEL — compare all 4 time periods for a city
# ============================================================

def get_best_time_comparison(city, current_period):

    if risk_df.empty:

        return [], None

    city = normalize_city(city)

    matches = risk_df[
        risk_df["city"].astype(str).str.strip().str.lower() == city
    ]

    rows = []

    for period in TIME_PERIOD_ORDER:

        match = matches[
            matches["time_period"].astype(str).str.strip().str.lower()
            == period.lower()
        ]

        if not match.empty:

            r = match.iloc[0]

            rows.append({
                "period": period,
                "risk_score": round(float(r["risk_score"]), 1),
                "risk_level": str(r["risk_level"]).title(),
                "is_current": period.lower() == current_period.lower(),
            })

    if not rows:
        return [], None

    best = min(rows, key=lambda r: r["risk_score"])

    return rows, best


# ============================================================
# ALL-PERIODS SNAPSHOT — full risk data for every time period
# of a city, in one payload. Powers the client-side time-slider
# so dragging it recolors the map/score instantly with no extra
# network round-trip, using only real precomputed data (no
# interpolation or invented numbers between periods).
# ============================================================

def get_all_periods_snapshot(city):

    snapshot = {}

    for period in TIME_PERIOD_ORDER:

        data = get_crime_data(city, period)

        if data is None:
            continue

        snapshot[period] = {
            "period": period,
            "score": data["score"],
            "risk_score": data["risk_score"],
            "risk_level": data["risk_level"],
            "risk_class": data["risk_class"],
            "map_color": data["map_color"],
            "summary": data["summary"],
        }

    return snapshot


# ============================================================
# ML INSIGHT — genuine model prediction, separate from the
# statistical risk score above.
# ============================================================

def get_ml_insight(city, current_time):

    if not ML_AVAILABLE:
        return None

    try:

        city_norm = normalize_city(city)

        if city_norm not in city_encoder.classes_:
            return None

        day_name = current_time.strftime("%A")

        if day_name not in day_encoder.classes_:
            return None

        city_enc = city_encoder.transform([city_norm])[0]
        day_enc = day_encoder.transform([day_name])[0]

        features = [[city_enc, current_time.hour, day_enc, current_time.month]]

        proba = ml_model.predict_proba(features)[0]
        classes = list(ml_model.classes_)

        if 1 not in classes:
            return None

        violent_prob = round(proba[classes.index(1)] * 100, 1)

        feature_labels = ["City", "Hour of day", "Day of week", "Month"]
        importances = dict(zip(feature_labels, ml_model.feature_importances_))
        top_feature = max(importances, key=importances.get)

        return {
            "violent_probability": violent_prob,
            "top_feature": top_feature,
        }

    except Exception as e:

        print("ML insight error:", e)
        return None


# ============================================================
# REAL NEARBY PLACES (Overpass API) — replaces fake offsets
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):

    R = 6371.0

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))


def get_nearby_amenity(lat, lng, tag_filter, label, radius=3000):

    """
    Query OpenStreetMap's Overpass API for real nearby places
    (hospitals, police stations, transit stops) and return the
    closest one, computed with an actual haversine distance —
    not a hardcoded offset from the user's location.
    """

    query = f"""
    [out:json][timeout:8];
    (
      node[{tag_filter}](around:{radius},{lat},{lng});
      way[{tag_filter}](around:{radius},{lat},{lng});
    );
    out center 10;
    """

    try:

        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers=HEADERS,
            timeout=9,
        )

        response.raise_for_status()

        elements = response.json().get("elements", [])

    except Exception as e:

        print(f"Overpass lookup failed ({label}):", e)
        return None

    best = None
    best_dist = None

    for el in elements:

        if "lat" in el and "lon" in el:
            elat, elon = el["lat"], el["lon"]

        elif "center" in el:
            elat, elon = el["center"]["lat"], el["center"]["lon"]

        else:
            continue

        name = el.get("tags", {}).get("name", label)
        dist = haversine_km(lat, lng, elat, elon)

        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = {
                "lat": elat,
                "lng": elon,
                "name": name,
                "distance_km": round(dist, 2),
                "found": True,
            }

    return best


def get_destinations(lat, lng):

    """
    Real hospital, police station and transit stop nearest the
    searched location. Falls back to a clearly-labelled demo
    point (found=False) only if the live lookup fails, so the
    UI never silently pretends a fake point is real.
    """

    hospital = get_nearby_amenity(lat, lng, '"amenity"="hospital"', "Nearest Hospital")

    police = get_nearby_amenity(lat, lng, '"amenity"="police"', "Nearest Police Station")

    transit = get_nearby_amenity(
        lat, lng, '"highway"="bus_stop"', "Nearest Bus Stop"
    ) or get_nearby_amenity(
        lat, lng, '"railway"="station"', "Nearest Station", radius=5000
    )

    fallback_offsets = {
        "hospital": (-0.022, 0.015, "Hospital (approximate — live lookup unavailable)"),
        "police": (0.009, 0.021, "Police Station (approximate — live lookup unavailable)"),
        "transit": (0.014, -0.010, "Transit Stop (approximate — live lookup unavailable)"),
    }

    def resolve(real, key, emoji, label):

        if real:
            return {
                "lat": real["lat"],
                "lng": real["lng"],
                "label": real["name"],
                "distance_km": real["distance_km"],
                "found": True,
                "emoji": emoji,
            }

        dlat, dlng, note = fallback_offsets[key]

        return {
            "lat": lat + dlat,
            "lng": lng + dlng,
            "label": note,
            "distance_km": None,
            "found": False,
            "emoji": emoji,
        }

    return {
        "hospital": resolve(hospital, "hospital", "🏥", "Hospital"),
        "police": resolve(police, "police", "🚔", "Police Station"),
        "transit": resolve(transit, "transit", "🚌", "Transit Stop"),
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# ANALYZE LOCATION
# ============================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    raw_input = request.form.get("location", "").strip()

    if not raw_input:

        return render_template(
            "index.html",
            error="Please enter a location.",
        )

    gps = parse_gps_input(raw_input)

    if gps:

        lat, lng = gps

        display_location = f"Your Location ({lat:.4f}, {lng:.4f})"

        geocoded = True

        city = reverse_geocode_city(lat, lng)

    else:

        (lat, lng, display_location, city, geocoded) = geocode_location(raw_input)

    if not geocoded:

        return render_template(
            "index.html",
            error=(
                "Could not find that location. "
                "Please try a more specific address or place name."
            ),
        )

    if not city:

        return render_template(
            "index.html",
            error="We found the location, but could not determine its city.",
        )

    current_time, time_period = get_current_time_period()

    data = get_crime_data(city, time_period)

    if data is None:

        return render_template(
            "index.html",
            error=(
                f"We found {display_location}, but SafePath-AI currently "
                f"does not have historical crime data for {city.title()}."
            ),
        )

    breakdown = get_risk_breakdown(data["stats"])
    best_time_rows, best_time = get_best_time_comparison(city, time_period)
    ml_insight = get_ml_insight(city, current_time)
    destinations = get_destinations(lat, lng)
    periods_snapshot = get_all_periods_snapshot(city)

    return render_template(
        "result.html",
        location=display_location,
        lat=lat,
        lng=lng,
        geocoded=geocoded,
        city=city.title(),
        time_period=time_period,
        current_time=current_time.strftime("%I:%M %p"),
        current_hour=current_time.hour,
        periods_snapshot=periods_snapshot,
        score=data["score"],
        risk_score=data["risk_score"],
        risk_level=data["risk_level"],
        risk_class=data["risk_class"],
        map_color=data["map_color"],
        summary=data["summary"],
        tips=data["tips"],
        stats=data["stats"],
        breakdown=breakdown,
        best_time_rows=best_time_rows,
        best_time=best_time,
        ml_insight=ml_insight,
        destinations=destinations,
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(debug=True)
