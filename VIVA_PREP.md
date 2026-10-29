# SafePath AI — Viva Prep

## One-paragraph project story

> Traditional navigation apps optimize for distance and time only. SafePath
> AI adds a safety dimension by combining geolocation, a real 40,000+ record
> historical Indian crime dataset, a statistical risk-scoring engine, a
> small supervised machine learning model, and real walking-route
> navigation — so a user can see not just how to get somewhere, but what
> historical crime patterns look like for that city and time of day, why
> the score looks the way it does, and when it's historically safer to
> travel.

## The two "AI" components — know the difference

**1. Statistical risk engine (`risk_engine.py`)**
A transparent formula: 60% crime frequency (relative to that city's own
average) + 40% average crime severity, producing a 0–100 risk score per
city + time period. Not machine learning — a weighted statistical model.
Be upfront about this if asked "is this AI?"

**2. Machine learning model (`train_model.py`)**
A Random Forest classifier trained on raw per-incident records
(city, hour, day of week, month → is the crime violent?). This is genuine
supervised learning: train/test split, evaluated with accuracy and
ROC-AUC, feature importances reported. This is what you point to when
asked "where's the ML?"

## Likely questions and strong answers

**Q: How do you know this specific route is safe?**
> We don't claim an individual street is safer than another. Our dataset
> has no incident-level coordinates, so street-level risk isn't something
> we can honestly calculate. The route line is colored by the city's
> overall historical risk for the current time period — that's stated
> directly in the UI. What we do provide accurately: a real walking route
> from OSRM, and a historically-grounded risk context for the area.

**Q: Where's the AI/ML?**
> Two places, and we keep them separate. The risk score is a transparent
> statistical formula, not ML. Separately, we trained a Random Forest
> classifier on the raw incident records to estimate the probability a
> crime at a given city, hour, day of week, and month is violent — that's
> the genuine ML component, with a held-out test set and reported
> accuracy/ROC-AUC.

**Q: Why Random Forest specifically?**
> It handles mixed categorical/numeric features well without heavy
> preprocessing, gives feature importances we can show the user
> (explainability matters for a safety app), and is robust to the noisy,
> imbalanced nature of crime data compared to something like logistic
> regression.

**Q: What are your model's limitations?**
> It's trained on historical, reported crime data — it reflects reporting
> patterns as much as actual incidence, it can't account for anything
> that changed after the dataset's collection period, and the class
> imbalance (most crimes aren't violent) means we used balanced class
> weights and still report the minority-class metrics, not just overall
> accuracy.

**Q: How did you get real hospital/police locations?**
> Live queries to OpenStreetMap's Overpass API around the searched
> coordinates, picking the nearest match by actual haversine distance.
> If that lookup fails (rate limit, no network), we fall back to a
> clearly-labeled approximate point rather than silently showing a fake
> one as real — the UI marks it "approximate" in that case.

**Q: Why not build the 1km×1km crime grid you mentioned wanting?**
> That would require incident-level coordinates we don't have. Randomly
> assigning coordinates to existing crimes to fake a grid would be
> fabricating data, and it's the first thing a careful examiner would
> probe. Future work with a coordinate-tagged dataset could genuinely
> support that.

## What NOT to say

- Don't call the statistical risk score "AI" or "machine learning."
- Don't imply the route coloring reflects per-street risk.
- Don't claim the hospital/police points are always exact — say what
  happens when the live lookup fails (approximate fallback, clearly
  labeled).
