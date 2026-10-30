# 🛡️ SafePath AI

A crime-risk-aware travel assistant. Enter a place in India and get:

- a **0–100 safety score** (animated ring) built from a real historical crime dataset
- **exactly why** — the 60 % frequency + 40 % severity recipe, and what makes this time of day different from the rest of the day
- the **best time to travel**, plus a **time-of-day slider** that recolours the map live
- a **machine-learning insight** — a Random Forest's hour-by-hour estimate, shown with its honest accuracy numbers
- **real nearby hospital / police station / transit stop** (OpenStreetMap) with real **walking routes** (OSRM)
- a personal **safety checklist** and **emergency numbers**
- a **printable report**, a shareable **link**, and **recent searches** (saved only in the visitor's browser)

Everything on screen comes from the dataset, the trained model, or OpenStreetMap. Nothing is invented — and where the data can't support a claim (street-level risk), the app says so.

---

## Project layout

```
SafePath-AI/
├── app.py                  Flask app (routes, lookups, caching, fallbacks)
├── city_utils.py           shared city-name normalisation (Bengaluru → bangalore …)
├── clean_kaggle.py         raw Kaggle CSV  → cleaned_crime_data.csv
├── risk_engine.py          cleaned data    → crime_risk_summary.csv   (statistical score)
├── train_model.py          cleaned data    → model/*.joblib + metrics.json  (ML)
├── explore_kaggle.py       optional: inspect the raw CSV's columns
├── requirements.txt
├── render.yaml             one-click Render deployment (+ health check)
├── .gitignore              (deliberately does NOT ignore CSVs or model/)
├── templates/              index.html · result.html · error.html
├── static/css/theme.css    shared design + print stylesheet
├── crimes_indian_dataset/  your data lives here
└── model/                  trained model lives here
```

## Run it locally (Windows / VS Code)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Put the raw Kaggle file at `crimes_indian_dataset/crime_dataset_india.csv`, then run the pipeline **in order**:

```
python clean_kaggle.py     # → crimes_indian_dataset/cleaned_crime_data.csv
python risk_engine.py      # → crimes_indian_dataset/crime_risk_summary.csv
python train_model.py      # → model/violent_predictor.joblib, encoders, metrics.json
python app.py              # → http://127.0.0.1:5000
```

**Already have `cleaned_crime_data.csv` and `crime_risk_summary.csv` from before?** Skip the first two steps — just run `python train_model.py` once (it now also writes `model/metrics.json`, which the page uses to show accuracy honestly) and then `python app.py`.

If you skip `train_model.py` the app still works; it just hides the ML card.

## Check that it's healthy

Open **`/health`** (locally `http://127.0.0.1:5000/health`, or on your live site). You should see:

```json
{"status": "ok", "cities": 29, "ml_model_loaded": true, "ml_metrics_loaded": true, ...}
```

`"status": "no-data"` means the server can't find `crimes_indian_dataset/crime_risk_summary.csv` — it wasn't committed.

## Deploy to Render (free)

1. Make sure these are committed and pushed to GitHub:
   `crimes_indian_dataset/cleaned_crime_data.csv`, `crimes_indian_dataset/crime_risk_summary.csv`
   (the model files are optional — Render re-trains the model during the build so it matches its own scikit-learn version).
2. Render → **New → Blueprint** → pick the repo. `render.yaml` sets everything up.
3. Wait for the build, then open `https://<your-service>.onrender.com/health`.

```
git add .
git commit -m "SafePath AI final version"
git push
```

**Free-tier notes:** the app sleeps after ~15 min idle and takes ~30–60 s to wake — open the link yourself a minute before a demo.

## How it stays reliable

| Problem | What the app does |
|---|---|
| Nominatim blocks/rate-limits the server | falls back to Photon, then to built-in city centres for plain city names |
| Geocoder returns a city under an odd key (e.g. Delhi) | checks every address field, then matches dataset city names in the text |
| Overpass is slow or down | tries a mirror; if both fail, markers are clearly labelled *approximate* |
| "Nearest hospital" being just *some* hospital | searches growing circles and picks the true nearest by haversine distance |
| Slow free-tier requests | lookups run in parallel, results are cached, gunicorn timeout raised |
| Browser Back button / refresh | results are a normal link (`/analyze?location=…`), so they refresh, share and bookmark |

## Known limitations (say these out loud — they're a strength)

- The dataset has **no incident coordinates**, so risk is **city × time-of-day**, not street-level. The route colour reflects the city's risk, never a single street.
- Scores compare **times of day within a city**. They are *not* for ranking cities against each other (frequency is measured against each city's own average).
- The ML model predicts *"is this incident violent?"* from only four features. Its ROC-AUC is shown on the page; if it's near 0.5 the page says it finds only weak patterns.
- The data is historical, reported crime from a public Kaggle dataset — not independently verified against official statistics.
