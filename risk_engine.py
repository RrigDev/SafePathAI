import pandas as pd
from pathlib import Path


# ============================================================
# SafePath-AI
# Real Crime Data Risk Engine
# ============================================================

INPUT_FILE = Path(
    "crimes_indian_dataset/cleaned_crime_data.csv"
)

OUTPUT_FILE = Path(
    "crimes_indian_dataset/crime_risk_summary.csv"
)


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

print("=" * 60)
print("SafePath-AI — REAL DATA RISK ENGINE")
print("=" * 60)

print("\nLoading cleaned crime dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df):,} crime records.")


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "city",
    "time_period",
    "crime_category",
    "crime_description"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR: Missing required columns:")

    for column in missing_columns:
        print(f"  - {column}")

    raise SystemExit(
        "\nRisk engine cannot continue."
    )


# ============================================================
# 3. DEFINE CRIME SEVERITY WEIGHTS
# ============================================================
#
# Higher weight = more serious contribution to risk.
#
# These are MODELING WEIGHTS, not official crime severity
# classifications.
#
# We keep them explicit so the model is transparent.
# ============================================================

severity_weights = {

    # Extremely serious violent crimes
    "HOMICIDE": 10,
    "SEXUAL ASSAULT": 9,
    "KIDNAPPING": 9,
    "FIREARM OFFENSE": 8,

    # Serious violent crimes
    "ASSAULT": 7,
    "ROBBERY": 7,
    "DOMESTIC VIOLENCE": 7,

    # Property crimes
    "ARSON": 6,
    "BURGLARY": 5,
    "VEHICLE - STOLEN": 5,
    "VANDALISM": 3,

    # Financial / cyber
    "EXTORTION": 5,
    "FRAUD": 4,
    "IDENTITY THEFT": 4,
    "CYBERCRIME": 4,
    "COUNTERFEITING": 3,

    # Theft
    "SHOPLIFTING": 2,

    # Public / drug
    "DRUG OFFENSE": 4,
    "ILLEGAL POSSESSION": 3,
    "PUBLIC INTOXICATION": 3,

    # Traffic
    "TRAFFIC VIOLATION": 1
}


# ============================================================
# 4. ASSIGN SEVERITY WEIGHT TO EVERY CRIME
# ============================================================

df["severity_weight"] = (
    df["crime_description"]
    .map(severity_weights)
    .fillna(1)
)


# ============================================================
# 5. GROUP CRIME DATA
# ============================================================
#
# We calculate risk for:
#
# CITY + TIME PERIOD
#
# Example:
#
# Bangalore + Night
# Bangalore + Morning
# Delhi + Night
# Mumbai + Evening
#
# ============================================================

group_columns = [
    "city",
    "time_period"
]

print("\nCalculating city/time statistics...")


summary = (
    df
    .groupby(group_columns)
    .agg(
        total_crimes=(
            "crime_description",
            "count"
        ),

        severity_points=(
            "severity_weight",
            "sum"
        ),

        violent_crimes=(
            "is_violent",
            "sum"
        ),

        theft=(
            "is_theft",
            "sum"
        ),

        robbery=(
            "is_robbery",
            "sum"
        ),

        vehicle_theft=(
            "is_vehicle_theft",
            "sum"
        ),

        property_crimes=(
            "is_property_crime",
            "sum"
        ),

        domestic_violence=(
            "is_domestic_violence",
            "sum"
        )
    )
    .reset_index()
)


# ============================================================
# 6. CALCULATE CITY BASELINES
# ============================================================
#
# Different cities have different numbers of records.
#
# We therefore compare each city/time combination against
# the normal crime level for that city.
#
# Example:
#
# Bangalore Night = 1157
#
# We don't simply say:
#
# "1157 = High"
#
# Instead we compare it with Bangalore's other time periods.
# ============================================================

city_baseline = (
    summary
    .groupby("city")["total_crimes"]
    .transform("mean")
)

summary["city_average_crimes"] = city_baseline


# ============================================================
# 7. CALCULATE CRIME FREQUENCY RATIO
# ============================================================

summary["frequency_ratio"] = (
    summary["total_crimes"]
    / summary["city_average_crimes"]
)


# ============================================================
# 8. CALCULATE SEVERITY PER CRIME
# ============================================================

summary["severity_per_crime"] = (
    summary["severity_points"]
    / summary["total_crimes"]
)


# ============================================================
# 9. NORMALIZE FREQUENCY
# ============================================================
#
# Frequency ratio around 1.0 means normal for that city.
#
# > 1.0 = more crimes than the city's average
#
# < 1.0 = fewer crimes than the city's average
#
# We convert this into a 0-100 scale.
# ============================================================

summary["frequency_score"] = (
    summary["frequency_ratio"] * 50
)

summary["frequency_score"] = (
    summary["frequency_score"]
    .clip(0, 100)
)


# ============================================================
# 10. NORMALIZE SEVERITY
# ============================================================

MAX_SEVERITY = 10

summary["severity_score"] = (
    summary["severity_per_crime"]
    / MAX_SEVERITY
) * 100

summary["severity_score"] = (
    summary["severity_score"]
    .clip(0, 100)
)


# ============================================================
# 11. COMBINE FREQUENCY + SEVERITY
# ============================================================
#
# Frequency = 60%
# Severity  = 40%
#
# This means frequent crime matters slightly more than the
# average severity of crimes.
# ============================================================

summary["risk_score"] = (
    summary["frequency_score"] * 0.60
    +
    summary["severity_score"] * 0.40
)


summary["risk_score"] = (
    summary["risk_score"]
    .clip(0, 100)
    .round(1)
)


# ============================================================
# 12. CLASSIFY RISK LEVEL
# ============================================================

def classify_risk(score):

    if score < 45:
        return "LOW"

    elif score < 55:
        return "MEDIUM"

    else:
        return "HIGH"


summary["risk_level"] = (
    summary["risk_score"]
    .apply(classify_risk)
)


# ============================================================
# 13. SORT RESULTS
# ============================================================

summary = summary.sort_values(
    ["city", "risk_score"],
    ascending=[True, False]
)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

summary.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 15. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RISK ENGINE COMPLETE")
print("=" * 60)

print(
    f"\nGenerated {len(summary):,} "
    "city/time combinations."
)

print("\nRisk level distribution:")

print(
    summary["risk_level"]
    .value_counts()
    .to_string()
)


print("\nTop 20 highest-risk combinations:")

print(
    summary[
        [
            "city",
            "time_period",
            "total_crimes",
            "violent_crimes",
            "theft",
            "robbery",
            "vehicle_theft",
            "risk_score",
            "risk_level"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


print("\n" + "=" * 60)

print("\nSaved risk summary to:")

print(OUTPUT_FILE)

print("\nNext step:")

print(
    "Connect this risk engine to the SafePath-AI Flask app."
)