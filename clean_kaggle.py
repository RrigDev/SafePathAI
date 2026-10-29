import pandas as pd
from pathlib import Path

# ============================================================
# SafePath-AI — Clean Indian Crime Dataset
# ============================================================

INPUT_FILE = Path("crimes_indian_dataset/crime_dataset_india.csv")
OUTPUT_FILE = Path("crimes_indian_dataset/cleaned_crime_data.csv")


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df):,} crime records.")


# ------------------------------------------------------------
# 2. Clean column names
# ------------------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


# ------------------------------------------------------------
# 3. Parse dates and times
# ------------------------------------------------------------

df["date_reported"] = pd.to_datetime(
    df["date_reported"],
    errors="coerce",
    dayfirst=True
)

df["date_of_occurrence"] = pd.to_datetime(
    df["date_of_occurrence"],
    errors="coerce",
    dayfirst=True
)

# Time of occurrence
# The dataset stores the time separately from the date.
# Time of occurrence
# Dataset format: DD-MM-YYYY HH:MM
df["time_of_occurrence"] = pd.to_datetime(
    df["time_of_occurrence"].astype(str).str.strip(),
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

# Extract hour
df["hour"] = df["time_of_occurrence"].dt.hour

# Extract additional date information
df["day_of_week"] = df["date_of_occurrence"].dt.day_name()
df["month"] = df["date_of_occurrence"].dt.month
df["year"] = df["date_of_occurrence"].dt.year


# ------------------------------------------------------------
# 4. Create time-of-day categories
# ------------------------------------------------------------

def get_time_period(hour):

    if pd.isna(hour):
        return "Unknown"

    hour = int(hour)

    if 5 <= hour < 12:
        return "Morning"

    elif 12 <= hour < 17:
        return "Afternoon"

    elif 17 <= hour < 21:
        return "Evening"

    else:
        return "Night"


df["time_period"] = df["hour"].apply(get_time_period)


# ------------------------------------------------------------
# 5. Standardize crime descriptions
# ------------------------------------------------------------

df["crime_description"] = (
    df["crime_description"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ------------------------------------------------------------
# 6. Create broader crime categories
# ------------------------------------------------------------

def categorize_crime(crime):

    # Theft-related crimes
    if crime in [
        "IDENTITY THEFT",
        "SHOPLIFTING"
    ]:
        return "THEFT"

    # Vehicle theft
    elif crime == "VEHICLE - STOLEN":
        return "VEHICLE THEFT"

    # Robbery
    elif crime == "ROBBERY":
        return "ROBBERY"

    # Violent crimes
    elif crime in [
        "ASSAULT",
        "HOMICIDE",
        "KIDNAPPING",
        "SEXUAL ASSAULT",
        "FIREARM OFFENSE"
    ]:
        return "VIOLENT CRIME"

    # Property crimes
    elif crime in [
        "BURGLARY",
        "VANDALISM",
        "ARSON"
    ]:
        return "PROPERTY CRIME"

    # Financial and cyber crimes
    elif crime in [
        "FRAUD",
        "CYBERCRIME",
        "COUNTERFEITING",
        "EXTORTION"
    ]:
        return "FINANCIAL / CYBER CRIME"

    # Public / drug-related crimes
    elif crime in [
        "DRUG OFFENSE",
        "ILLEGAL POSSESSION",
        "PUBLIC INTOXICATION"
    ]:
        return "PUBLIC / DRUG OFFENSE"

    # Traffic
    elif crime == "TRAFFIC VIOLATION":
        return "TRAFFIC"

    # Domestic violence
    elif crime == "DOMESTIC VIOLENCE":
        return "DOMESTIC VIOLENCE"

    # Anything unexpected
    else:
        return "OTHER"


df["crime_category"] = df["crime_description"].apply(
    categorize_crime
)


# ------------------------------------------------------------
# 7. Create individual risk indicators
# ------------------------------------------------------------

df["is_theft"] = (
    df["crime_category"] == "THEFT"
).astype(int)

df["is_robbery"] = (
    df["crime_category"] == "ROBBERY"
).astype(int)

df["is_vehicle_theft"] = (
    df["crime_category"] == "VEHICLE THEFT"
).astype(int)

df["is_violent"] = (
    df["crime_category"] == "VIOLENT CRIME"
).astype(int)

df["is_property_crime"] = (
    df["crime_category"] == "PROPERTY CRIME"
).astype(int)

df["is_domestic_violence"] = (
    df["crime_category"] == "DOMESTIC VIOLENCE"
).astype(int)


# ------------------------------------------------------------
# 8. Clean categorical fields
# ------------------------------------------------------------

categorical_columns = [
    "city",
    "victim_gender",
    "weapon_used",
    "crime_domain",
    "case_closed"
]

for column in categorical_columns:

    if column in df.columns:

        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )


# ------------------------------------------------------------
# 9. Handle numeric values
# ------------------------------------------------------------

df["victim_age"] = pd.to_numeric(
    df["victim_age"],
    errors="coerce"
)

df["police_deployed"] = pd.to_numeric(
    df["police_deployed"],
    errors="coerce"
)


# ------------------------------------------------------------
# 10. Remove duplicate records
# ------------------------------------------------------------

before = len(df)

df = df.drop_duplicates()

removed = before - len(df)

print(f"Removed {removed:,} duplicate records.")


# ------------------------------------------------------------
# 11. Select final columns
# ------------------------------------------------------------

final_columns = [
    "report_number",
    "date_reported",
    "date_of_occurrence",
    "city",
    "crime_code",
    "crime_description",
    "crime_category",
    "crime_domain",
    "hour",
    "time_period",
    "day_of_week",
    "month",
    "year",
    "victim_age",
    "victim_gender",
    "weapon_used",
    "police_deployed",
    "case_closed",
    "is_theft",
    "is_robbery",
    "is_vehicle_theft",
    "is_violent",
    "is_property_crime",
    "is_domestic_violence"
]

df = df[
    [column for column in final_columns if column in df.columns]
]


# ------------------------------------------------------------
# 12. Save cleaned dataset
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 13. Print summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLEANING COMPLETE")
print("=" * 60)

print(
    f"\nFinal shape: "
    f"{df.shape[0]:,} rows × {df.shape[1]} columns"
)

print("\nCrime categories:")
print(
    df["crime_category"]
    .value_counts()
    .to_string()
)

print("\nCities:")
print(
    df["city"]
    .value_counts()
    .head(10)
    .to_string()
)

print("\nTime periods:")
print(
    df["time_period"]
    .value_counts()
    .to_string()
)

print("\nMissing hours:")
print(
    f"{df['hour'].isna().sum():,} records"
)

print("\nSaved cleaned dataset to:")
print(OUTPUT_FILE)

print("\nNext step:")
print(
    "Use cleaned_crime_data.csv "
    "for SafePath-AI analysis/modeling."
)
