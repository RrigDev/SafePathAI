# SafePath AI — Viva Prep

## 30-second pitch

> Navigation apps optimise for distance and time. SafePath AI adds a safety dimension. It combines a real
> historical crime dataset, a transparent statistical risk score, a small machine-learning model, and real
> map data. You enter a place and get a safety score, the exact reasons behind it, the historically safest
> time of day, and a real walking route to the nearest hospital or police station.
> We are explicit about what the data can't tell us — it has no street-level coordinates, so we never
> claim one street is safer than another.

## The two "AI" components — know the difference

**1. Statistical risk engine (`risk_engine.py`) — transparent formula, not ML**
For every city + time band: `risk = 0.60 × frequency score + 0.40 × severity score`.
- *Frequency*: incidents in that band ÷ the city's own average per band (×50, capped at 100).
- *Severity*: average crime-type weight (homicide 10 … traffic violation 1) ÷ 10 × 100.
- *Safety score* = 100 − risk. Levels: < 45 low, < 55 medium, otherwise high.
The weights are **our modelling choices**, not official classifications. The results page shows this
recipe with the real numbers, so the score is never a black box.

**2. Machine-learning model (`train_model.py`) — genuine supervised learning**
Random Forest classifier on the raw incident records. Features: city, hour, weekday, month. Target: is the
incident violent (0/1). 80/20 stratified split. The app asks it for all 24 hours and draws the curve.

## Hard questions — and honest answers

**"Where is the AI?"**
Two parts, kept separate on purpose. The score is a transparent weighted formula. The ML part is a trained
Random Forest with a held-out test set, and the app displays its real accuracy and ROC-AUC.

**"How accurate is your model?"**
Be direct: *"Violent crimes are a minority, so plain accuracy is misleading — predicting 'never violent'
already scores about as well. That's why we report ROC-AUC and the baseline too. With only city, hour,
weekday and month as inputs the model finds modest-to-weak patterns, and the page says so instead of
overselling it."* Examiners rate honesty about weak results above inflated claims.

**"Why didn't you use `class_weight='balanced'`?"**
Balancing inflates predicted probabilities, so "X % violent" would no longer mean X %. We want the
probabilities to stay close to real frequencies because we show them to users.

**"Why a Random Forest?"**
Handles categorical + numeric tabular features with little preprocessing, resists overfitting, and gives
feature importances for explainability. Depth and leaf size are limited (`max_depth=8`, `min_samples_leaf=25`)
to keep it from memorising noise.

**"How do you know this route is safe?"**
We don't claim that. The route is a real walking route (OSRM). Its colour reflects the *city's* historical
risk for that time band. The dataset has no incident coordinates, so street-level risk can't honestly be
computed — and faking it by assigning random coordinates would be fabricating data.

**"Why not rank the cities from safest to riskiest?"**
Frequency is measured against *each city's own* average, so scores compare times of day within a city.
Ranking cities by them would be comparing different baselines (and ignoring population). We deliberately
left that out and note it on the page.

**"Where does the data come from, and is it reliable?"**
A public Kaggle dataset of Indian crime records (~40,000 rows). It is historical, reported crime only, and we
haven't independently verified it against official sources — that is a stated limitation.

**"Where do the hospitals / police stations come from?"**
OpenStreetMap through the Overpass API. We search growing circles and pick the true nearest by haversine
distance. If the service is unreachable, the marker is labelled *approximate* instead of pretending to be real.

**"What about privacy?"**
No accounts, no database of users. Recent searches and checklist ticks live only in the visitor's own browser
(`localStorage`). The place text is sent to OpenStreetMap geocoders to find coordinates.

**"What happens if a service goes down?"**
Geocoding falls back Nominatim → Photon → built-in city centres; Overpass has a mirror; results are cached;
failures show friendly messages rather than errors. `/health` shows whether data and model loaded.

**"What would you do next?"**
Incident-level coordinates (→ real spatial grid and risk-aware routing), per-capita normalisation (→ fair
city comparison), more features for the model (crime type, victim profile), probability calibration,
and evaluation on a later time period than training.

## What NOT to say

- Don't call the risk score "machine learning".
- Don't imply the route colour is street-level risk.
- Don't quote the model's probability as a prediction of a crime happening to the user.
- Don't claim the hospital / police points are always exact — say what happens when lookups fail.

## 60-second demo script

1. Homepage → click **Bangalore**.
2. Point at the **score ring** and the report sentence.
3. **Why this score** → show 60 / 40 split and the real numbers.
4. **Best time** → drag the **clock slider**; the map and chart recolour.
5. **ML insight** → show the curve, then the honesty check (ROC-AUC).
6. **Route finder** → click *Nearest hospital*.
7. **Print report** → clean one-page PDF.
8. Finish on the **Methodology** card: "and here are the limitations."
