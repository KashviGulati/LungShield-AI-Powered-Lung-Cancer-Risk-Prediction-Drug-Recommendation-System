from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
with open("cancer_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load drug data
df = pd.read_csv(r"D:\Cancer risk prediction\dataset\Integrated_Lung_Cancer_Data.csv")

# Define expected features and valid ranges (based on dataset)
FEATURES = ['AGE', 'GENDER', 'SMOKING', 'ANXIETY', 'FATIGUE', 'COUGHING']
VALID_RANGES = {
    'AGE': (20, 80),  # Based on dataset min/max (21-76, rounded)
    'GENDER': (1, 2),
    'SMOKING': (1, 2),
    'ANXIETY': (1, 2),
    'FATIGUE': (1, 2),
    'COUGHING': (1, 2)
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log all form data
        print("📋 All Form Data:", dict(request.form))

        # Collect and validate features
        features = []
        for feature in FEATURES:
            value = request.form.get(feature)
            if value is None:
                raise ValueError(f"Missing feature: {feature}")
            value = float(value)
            min_val, max_val = VALID_RANGES[feature]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{feature} value {value} out of range [{min_val}, {max_val}]")
            features.append(value)
        print("🔍 Raw Features:", features)

        # Scale and predict
        scaled_features = scaler.transform([features])
        print("📏 Scaled Features:", scaled_features)
        prediction = model.predict(scaled_features)[0]
        print("🧠 Prediction:", prediction)

        # Result logic
        if prediction == 1:
            drug = df[df['LUNG_CANCER'] == 1]['Recommended Drug'].sample(1).values[0]
            risk = "High"
        else:
            drug = "No drug needed"
            risk = "Low"

        return render_template("result.html", risk=risk, drug=drug)

    except ValueError as ve:
        return f"Input Error: {str(ve)}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)