from flask import Flask, render_template, request
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

app = Flask(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

GEOCODE_URL = "https://nominatim.openstreetmap.org/search"
REVERSE_GEOCODE_URL = "https://nominatim.openstreetmap.org/reverse"

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
# CITY NAME NORMALIZATION
# ============================================================

def normalize_city(city):

    if not city:
        return ""

    city = city.strip().lower()

    # Nominatim may return Bengaluru,
    # while our dataset uses Bangalore.
    city_aliases = {

        "bengaluru": "bangalore",
        "bengaluru urban": "bangalore",

        "bombay": "mumbai",

        "calcutta": "kolkata",

        "madras": "chennai",

        "baroda": "ahmedabad",

    }

    return city_aliases.get(city, city)


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

            # Nominatim can store a city under different keys
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

    """
    Convert GPS coordinates into a city name.

    Used when the user presses "Use My Location".
    """

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

    """
    Parse raw GPS input:

        12.9716,77.5946

    Returns:

        (lat, lng)

    or:

        None
    """

    if "," not in location_str:

        return None

    parts = location_str.split(",")

    if len(parts) != 2:

        return None

    try:

        lat = float(parts[0].strip())
        lng = float(parts[1].strip())

        if (
            -90 <= lat <= 90
            and
            -180 <= lng <= 180
        ):

            return lat, lng

    except ValueError:

        return None

    return None


# ============================================================
# CURRENT TIME PERIOD
# ============================================================

def get_current_time_period():

    """
    Get the current time in India and convert it
    into the same time periods used by our dataset.
    """

    now = datetime.now(
        INDIA_TIMEZONE
    )

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

    IMPORTANT:

    This is city + time based historical risk.

    It is NOT street-level crime prediction because
    the source dataset does not contain crime coordinates.
    """

    if risk_df.empty:

        return None

    city = normalize_city(city)

    matches = risk_df[
        (
            risk_df["city"]
            .astype(str)
            .str.strip()
            .str.lower()
            == city
        )
        &
        (
            risk_df["time_period"]
            .astype(str)
            .str.strip()
            .str.lower()
            == time_period.lower()
        )
    ]

    if matches.empty:

        return None

    row = matches.iloc[0]

    risk_score = float(
        row["risk_score"]
    )

    risk_level = str(
        row["risk_level"]
    ).upper()


    # --------------------------------------------------------
    # Convert risk score into a user-friendly SAFETY score
    #
    # Risk score:
    #     Higher = more dangerous
    #
    # Safety score:
    #     Higher = safer
    # --------------------------------------------------------

    safety_score = round(
        max(
            0,
            min(
                100,
                100 - risk_score
            )
        )
    )


    # --------------------------------------------------------
    # UI classes
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Return data for frontend
    # --------------------------------------------------------

    return {

        "score": safety_score,

        "risk_score": round(
            risk_score,
            1
        ),

        "risk_level": risk_level.title(),

        "risk_class": risk_class,

        "map_color": map_color,

        "summary": summary,

        "tips": tips,

        "city": str(
            row["city"]
        ),

        "time_period": str(
            row["time_period"]
        ),

        "stats": {

            "total_crimes": int(
                row["total_crimes"]
            ),

            "violent_crimes": int(
                row["violent_crimes"]
            ),

            "theft": int(
                row["theft"]
            ),

            "robbery": int(
                row["robbery"]
            ),

            "vehicle": int(
                row["vehicle_theft"]
            ),

            "property_crimes": int(
                row["property_crimes"]
            ),

            "domestic_violence": int(
                row["domestic_violence"]
            ),
        }
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# ANALYZE LOCATION
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    raw_input = request.form.get(
        "location",
        ""
    ).strip()


    # --------------------------------------------------------
    # Empty input
    # --------------------------------------------------------

    if not raw_input:

        return render_template(

            "index.html",

            error=(
                "Please enter a location."
            )
        )


    # --------------------------------------------------------
    # Check if GPS coordinates
    # --------------------------------------------------------

    gps = parse_gps_input(
        raw_input
    )


    if gps:

        lat, lng = gps

        display_location = (
            f"Your Location "
            f"({lat:.4f}, {lng:.4f})"
        )

        geocoded = True

        # Find city from GPS
        city = reverse_geocode_city(
            lat,
            lng
        )

    else:

        # ----------------------------------------------------
        # Normal text location
        # ----------------------------------------------------

        (
            lat,
            lng,
            display_location,
            city,
            geocoded
        ) = geocode_location(
            raw_input
        )


    # --------------------------------------------------------
    # If geocoding failed
    # --------------------------------------------------------

    if not geocoded:

        return render_template(

            "index.html",

            error=(
                "Could not find that location. "
                "Please try a more specific address "
                "or place name."
            )
        )


    # --------------------------------------------------------
    # Check city
    # --------------------------------------------------------

    if not city:

        return render_template(

            "index.html",

            error=(
                "We found the location, but "
                "could not determine its city."
            )
        )


    # --------------------------------------------------------
    # Current Indian time
    # --------------------------------------------------------

    current_time, time_period = (
        get_current_time_period()
    )


    # --------------------------------------------------------
    # REAL DATA LOOKUP
    # --------------------------------------------------------

    data = get_crime_data(
        city,
        time_period
    )


    # --------------------------------------------------------
    # Dataset does not cover this city
    # --------------------------------------------------------

    if data is None:

        return render_template(

            "index.html",

            error=(
                f"We found {display_location}, "
                f"but SafePath-AI currently does "
                f"not have historical crime data "
                f"for {city.title()}."
            )
        )


    # --------------------------------------------------------
    # Render result
    # --------------------------------------------------------

    return render_template(

        "result.html",

        location=display_location,

        lat=lat,

        lng=lng,

        geocoded=geocoded,

        city=city.title(),

        time_period=time_period,

        current_time=current_time.strftime(
            "%I:%M %p"
        ),

        score=data["score"],

        risk_score=data["risk_score"],

        risk_level=data["risk_level"],

        risk_class=data["risk_class"],

        map_color=data["map_color"],

        summary=data["summary"],

        tips=data["tips"],

        stats=data["stats"],
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )