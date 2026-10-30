"""
train_model.py

SafePath-AI — Machine Learning component.

risk_engine.py builds a transparent STATISTICAL summary
    (city + time_period -> aggregated risk_score).

This script trains a separate, genuinely supervised ML model on the RAW
per-incident records:

    features : city, hour of day, day of week, month
    target   : is the incident a violent crime?  (1 / 0)
    model    : Random Forest classifier

At runtime the app asks the model: "for this city, on this day of the week
and month, what share of incidents at each hour is estimated to be violent?"
and draws that as a 24-hour curve.

Design choices worth knowing (and saying in a viva):

  * No class_weight="balanced". Balancing inflates predicted probabilities,
    which would make "X% violent" misleading. Without it the probabilities
    stay close to real frequencies, so they can be shown to a user.
  * Because "violent" is a minority class, plain accuracy is a weak metric
    (predicting "never violent" already scores well). We therefore also
    report the majority-class baseline and ROC-AUC (0.5 = no better than
    chance). These are saved to model/metrics.json and shown honestly in
    the app.

Run order:

    python clean_kaggle.py
    python risk_engine.py
    python train_model.py
    python app.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from city_utils import normalize_city


DATA_FILE = Path("crimes_indian_dataset/cleaned_crime_data.csv")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["city_enc", "hour", "day_enc", "month"]
FEATURE_LABELS = {
    "city_enc": "City",
    "hour": "Hour of day",
    "day_enc": "Day of week",
    "month": "Month",
}


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
print(f"Dropped {before - len(df):,} rows with missing hour/day/month/city.")


# ------------------------------------------------------------
# 3. Normalise city so training matches the app's lookups
# ------------------------------------------------------------

df["city_norm"] = df["city"].apply(normalize_city)


# ------------------------------------------------------------
# 4. Encode categorical features
# ------------------------------------------------------------

city_encoder = LabelEncoder()
day_encoder = LabelEncoder()

df["city_enc"] = city_encoder.fit_transform(df["city_norm"])
df["day_enc"] = day_encoder.fit_transform(df["day_of_week"].astype(str))
df["hour"] = df["hour"].astype(int)
df["month"] = df["month"].astype(int)

X = df[FEATURES]
y = df["is_violent"].astype(int)

base_rate = float(y.mean())

print(f"\nViolent share of all incidents: {base_rate * 100:.1f}%")


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
    max_depth=8,
    min_samples_leaf=25,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)


# ------------------------------------------------------------
# 7. Evaluate honestly
# ------------------------------------------------------------

proba = model.predict_proba(X_test)[:, 1]
preds = (proba >= 0.5).astype(int)

accuracy = float(accuracy_score(y_test, preds))
baseline_accuracy = float(1 - y_test.mean())  # always predict "not violent"

try:
    auc = float(roc_auc_score(y_test, proba))
except ValueError:
    auc = None

print(f"\nTest accuracy          : {accuracy:.3f}")
print(f"Majority-class baseline: {baseline_accuracy:.3f}  (always predict 'not violent')")

if auc is not None:
    print(f"Test ROC-AUC           : {auc:.3f}  (0.5 = no better than chance)")


# ------------------------------------------------------------
# 8. Feature importance
# ------------------------------------------------------------

importances = {
    FEATURE_LABELS[name]: round(float(value), 4)
    for name, value in zip(FEATURES, model.feature_importances_)
}

print("\nFeature importances:")
for label, value in sorted(importances.items(), key=lambda kv: -kv[1]):
    print(f"  {label:15s}: {value:.3f}")


# ------------------------------------------------------------
# 9. Save model, encoders and metrics
# ------------------------------------------------------------

joblib.dump(model, MODEL_DIR / "violent_predictor.joblib")
joblib.dump(city_encoder, MODEL_DIR / "city_encoder.joblib")
joblib.dump(day_encoder, MODEL_DIR / "day_encoder.joblib")

metrics = {
    "model": "Random Forest classifier",
    "target": "incident is a violent crime",
    "features": list(FEATURE_LABELS.values()),
    "n_records": int(len(df)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "violent_base_rate_pct": round(base_rate * 100, 1),
    "accuracy": round(accuracy, 3),
    "baseline_accuracy": round(baseline_accuracy, 3),
    "roc_auc": round(auc, 3) if auc is not None else None,
    "feature_importances": importances,
    "sklearn_version": sklearn.__version__,
    "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
}

(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

print("\n" + "=" * 60)
print("Saved model + metrics to model/")
print("=" * 60)
print("\nNext step: start app.py — it loads these files automatically.")
