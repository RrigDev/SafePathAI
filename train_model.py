"""
train_model.py

SafePath-AI — Machine Learning component.

This is deliberately kept separate from risk_engine.py.

risk_engine.py builds a STATISTICAL summary
    (city + time_period -> aggregated risk_score).

train_model.py trains an actual supervised ML model
    (Random Forest classifier) on the RAW per-incident
    records to predict something the summary table can't
    give you directly: given a city, hour, day of week and
    month, what is the estimated probability that a crime
    occurring under those conditions is violent in nature?

This keeps a clean separation you can explain in a viva:

    "Our risk score is a transparent statistical formula.
     Our ML model is a separate, genuinely trained
     classifier that supplements it with a pattern-based
     estimate, not just a lookup."

Run this once after clean_kaggle.py, before starting app.py:

    python clean_kaggle.py
    python risk_engine.py
    python train_model.py
    python app.py
"""

import pandas as pd
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from city_utils import normalize_city


DATA_FILE = Path("crimes_indian_dataset/cleaned_crime_data.csv")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)


print("=" * 60)
print("SafePath-AI — ML MODEL TRAINING")
print("=" * 60)

print("\nLoading cleaned crime dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Loaded {len(df):,} crime records.")


# ------------------------------------------------------------
# 1. Required columns
# ------------------------------------------------------------

required = ["city", "hour", "day_of_week", "month", "is_violent"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise SystemExit(
        f"ERROR: cleaned_crime_data.csv is missing columns: {missing}\n"
        "Run clean_kaggle.py first."
    )


# ------------------------------------------------------------
# 2. Drop rows with missing feature values
# ------------------------------------------------------------

before = len(df)
df = df.dropna(subset=["hour", "day_of_week", "month", "city"])
print(f"\nDropped {before - len(df):,} rows with missing hour/day/month/city.")


# ------------------------------------------------------------
# 3. Normalize city so training matches the app's lookups
# ------------------------------------------------------------

df["city_norm"] = df["city"].apply(normalize_city)


# ------------------------------------------------------------
# 4. Encode categorical features
# ------------------------------------------------------------

city_encoder = LabelEncoder()
day_encoder = LabelEncoder()

df["city_enc"] = city_encoder.fit_transform(df["city_norm"])
df["day_enc"] = day_encoder.fit_transform(df["day_of_week"].astype(str))

FEATURES = ["city_enc", "hour", "day_enc", "month"]
TARGET = "is_violent"

X = df[FEATURES]
y = df[TARGET].astype(int)

print(f"\nClass balance (is_violent): \n{y.value_counts(normalize=True).round(3).to_string()}")


# ------------------------------------------------------------
# 5. Train / test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ------------------------------------------------------------
# 6. Train Random Forest
# ------------------------------------------------------------

print("\nTraining Random Forest classifier...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)


# ------------------------------------------------------------
# 7. Evaluate
# ------------------------------------------------------------

preds = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, preds)

print(f"\nTest accuracy: {acc:.3f}")

try:
    auc = roc_auc_score(y_test, proba)
    print(f"Test ROC-AUC:  {auc:.3f}")
except ValueError:
    pass

print("\nClassification report:")
print(classification_report(y_test, preds, target_names=["Not violent", "Violent"]))


# ------------------------------------------------------------
# 8. Feature importance (useful for the "Why?" explanation
#    and for the viva)
# ------------------------------------------------------------

feature_labels = {
    "city_enc": "City",
    "hour": "Hour of day",
    "day_enc": "Day of week",
    "month": "Month",
}

importances = dict(zip(FEATURES, model.feature_importances_))

print("\nFeature importances:")
for k, v in sorted(importances.items(), key=lambda kv: -kv[1]):
    print(f"  {feature_labels[k]:15s}: {v:.3f}")


# ------------------------------------------------------------
# 9. Save model + encoders
# ------------------------------------------------------------

joblib.dump(model, MODEL_DIR / "violent_predictor.joblib")
joblib.dump(city_encoder, MODEL_DIR / "city_encoder.joblib")
joblib.dump(day_encoder, MODEL_DIR / "day_encoder.joblib")

print("\n" + "=" * 60)
print("Saved model files to model/")
print("=" * 60)
print("\nNext step: start app.py — it will auto-load this model")
print("and show an ML Insight panel on the results page.")
