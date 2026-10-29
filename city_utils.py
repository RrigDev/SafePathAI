# ============================================================
# SafePath-AI — Shared city-name normalization
#
# Both app.py (risk lookup) and train_model.py (ML training)
# import this so a city is always spelled the same way on
# both sides. Mismatched spelling was a real bug risk before:
# Nominatim returns "Bengaluru", the Kaggle dataset uses
# "Bangalore" in some rows and "Bengaluru" in others.
# ============================================================

CITY_ALIASES = {
    "bengaluru": "bangalore",
    "bengaluru urban": "bangalore",
    "bombay": "mumbai",
    "calcutta": "kolkata",
    "madras": "chennai",
    "baroda": "ahmedabad",
}


def normalize_city(city):
    """Lowercase, strip, and collapse known aliases to one canonical name."""
    if not city:
        return ""

    city = str(city).strip().lower()

    return CITY_ALIASES.get(city, city)
