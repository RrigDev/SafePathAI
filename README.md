# SafePath AI

A crime-risk-aware navigation web app: geocodes a location, shows historical
city-level crime risk for the current time period, explains *why* that score
looks the way it does, compares risk across times of day, gives a small
genuine ML prediction, and draws a real walking route to the nearest real
hospital / police station / transit stop.

## What changed in this pass

1. **Backend ↔ template mismatch fixed.** `result.html` now uses the exact
   field names `app.py` sends (`total_crimes`, `violent_crimes`, `theft`,
   `robbery`, `vehicle`, `property_crimes`, `domestic_violence`).
2. **Route coloring is now honest.** The old code faked a sine-wave color
   gradient along the route to *look* like street-level risk data. That's
   gone. The route is now a single color tied to the city's actual
   historical risk level, and the UI says so explicitly.
3. **Real hospital / police / transit lookups.** The old hardcoded lat/lng
   offsets (`CENTER_LAT + 0.018`, etc.) are replaced with live queries to
   OpenStreetMap's Overpass API, picking the actual nearest match by
   haversine distance. If a live lookup fails, the UI clearly marks the
   point as approximate — it never silently shows a fake point as real.
4. **"Why is this area risky?"** — a ranked, percentage breakdown of the
   crime categories behind the score, built from data you already have.
5. **"Best time to travel"** — compares all four time periods for the same
   city and highlights the historically lowest-risk one.
6. **A real ML component** (`train_model.py`) — a Random Forest classifier
   trained on the raw per-incident data (not the aggregated summary) to
   estimate the probability a crime at a given city/hour/day/month is
   violent. This is a separate, genuinely trained model, clearly described
   as supplementing — not replacing — the statistical risk score.
7. **Fixed `requirements.txt`** — it was missing `pandas` entirely, which
   would have crashed the app on Render. Added `pandas`, `scikit-learn`,
   `joblib`.

## Project structure

```
SafePath-AI/
├── app.py                  # Flask app
├── city_utils.py           # shared city-name normalization
├── clean_kaggle.py         # raw Kaggle CSV -> cleaned_crime_data.csv
├── explore_kaggle.py       # inspect the raw Kaggle CSV columns
├── risk_engine.py          # cleaned data -> crime_risk_summary.csv (statistical model)
├── train_model.py          # cleaned data -> ML model (Random Forest)
├── requirements.txt
├── render.yaml
├── templates/
│   ├── index.html
│   └── result.html
├── crimes_indian_dataset/  # put your Kaggle CSV here (see below)
└── model/                  # train_model.py writes the .joblib files here
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the Kaggle "Indian Crimes" dataset and place the raw CSV at:

```
crimes_indian_dataset/crime_dataset_india.csv
```

Then run the pipeline **in this order**:

```bash
python explore_kaggle.py     # optional — sanity-check the columns
python clean_kaggle.py       # -> crimes_indian_dataset/cleaned_crime_data.csv
python risk_engine.py        # -> crimes_indian_dataset/crime_risk_summary.csv
python train_model.py        # -> model/violent_predictor.joblib + encoders
python app.py                # runs at http://127.0.0.1:5000
```

If you skip `train_model.py`, the app still runs fine — it just won't show
the ML Insight panel (checked at startup, logged to the console, no crash).

## Deploying to Render

1. Push this repo to GitHub, **including** `crimes_indian_dataset/*.csv` and
   `model/*.joblib` (Render doesn't run your local pipeline for you — commit
   the generated files, or add a build step that runs the pipeline).
2. On Render: New → Web Service → connect the repo. `render.yaml` already
   sets the build/start commands (`gunicorn app:app`).
3. First deploy will take a minute; subsequent pushes to your connected
   branch auto-redeploy.

## Known limitations (say these out loud in your viva — it's a strength, not a weakness)

- The dataset has **no per-incident latitude/longitude**, so risk is
  city + time-period level, not street level. The route color reflects
  this honestly — one color per route, not a fake gradient.
- Overpass and OSRM are free public instances with rate limits. Fine for a
  live demo; not meant for production traffic.
- The ML model predicts "is this crime violent," a real classification
  task on real features — it doesn't (and shouldn't be described as)
  predicting real-time crime.
