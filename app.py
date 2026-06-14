from flask import Flask, render_template, request
import random

app = Flask(__name__)

def get_crime_data(location):
    """
    Simulated crime data engine.
    Later you'll replace this with your actual ML model.
    """
    # Simulate different risk profiles based on location name length (placeholder logic)
    seed = sum(ord(c) for c in location.lower())
    random.seed(seed)

    theft      = random.randint(10, 80)
    robbery    = random.randint(2, 25)
    vehicle    = random.randint(5, 40)
    harassment = random.randint(3, 20)

    total = theft + robbery + vehicle + harassment
    score = max(10, 100 - total // 2)

    if score >= 70:
        risk_level = "Low"
        risk_class = "risk-low"
        map_color  = "#28a745"
        summary    = "This area is generally safe. Stay alert as usual."
        tips = [
            "✅ Area is well-lit and patrolled",
            "🚶 Walking at night is relatively safe",
            "📱 Keep emergency contacts handy",
            "🔒 Lock your vehicle when parked"
        ]
    elif score >= 45:
        risk_level = "Medium"
        risk_class = "risk-medium"
        map_color  = "#ffc107"
        summary    = "Moderate risk detected. Exercise caution."
        tips = [
            "⚠️ Avoid isolated roads after dark",
            "🚌 Prefer public transport at night",
            "👥 Travel in groups when possible",
            "💡 Stay in well-lit areas"
        ]
    else:
        risk_level = "High"
        risk_class = "risk-high"
        map_color  = "#dc3545"
        summary    = "High risk area. Take extra precautions."
        tips = [
            "🚨 Avoid this area after dark",
            "📞 Share your location with someone",
            "🏃 Do not walk alone",
            "🚖 Use trusted cab services only"
        ]

    return {
        "score": score,
        "risk_level": risk_level,
        "risk_class": risk_class,
        "map_color": map_color,
        "summary": summary,
        "tips": tips,
        "stats": {
            "theft": theft,
            "robbery": robbery,
            "vehicle": vehicle,
            "harassment": harassment
        }
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    location = request.form.get('location')

    # Handle GPS coordinates vs text input
    if ',' in location and location.replace(',','').replace('.','').replace('-','').strip().isdigit() == False:
        try:
            parts = location.split(',')
            lat = float(parts[0].strip())
            lng = float(parts[1].strip())
            display_location = f"{lat:.4f}, {lng:.4f}"
        except:
            lat, lng = 12.9716, 77.5946  # Default: Bangalore
            display_location = location
    else:
        lat, lng = 12.9716, 77.5946  # Default coordinates for text searches
        display_location = location

    data = get_crime_data(location)

    return render_template('result.html',
        location=display_location,
        lat=lat,
        lng=lng,
        score=data['score'],
        risk_level=data['risk_level'],
        risk_class=data['risk_class'],
        map_color=data['map_color'],
        summary=data['summary'],
        tips=data['tips'],
        stats=data['stats']
    )

if __name__ == '__main__':
    app.run(debug=True)