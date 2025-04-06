

from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

with open("d:/Cancer risk prediction/cancer_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("d:/Cancer risk prediction/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


df = pd.read_csv("d:/Cancer risk prediction/dataset/Integrated_Lung_Cancer_Data.csv")


moderate_drugs = ["Budesonide"]
high_drugs = ["Cisplatin", "Carboplatin", "Paclitaxel"]

FEATURES = ['AGE', 'GENDER', 'SMOKING', 'FATIGUE', 'WHEEZING', 'COUGHING']
VALID_RANGES = {
    'AGE': (20, 80),
    'GENDER': (1, 2),
    'SMOKING': (1, 2),
    'FATIGUE': (1, 2),
    'WHEEZING': (1, 2),
    'COUGHING': (1, 2)
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("📋 All Form Data:", dict(request.form))

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

        scaled_features = scaler.transform([features])
        print("📏 Scaled Features:", scaled_features)
        prob = model.predict_proba(scaled_features)[0][1]
        print("🧠 Probability of High Risk:", prob)

        if prob < 0.7:
            risk = "Low"
            drug = "No drug needed"
        elif prob < 0.9:
            risk = "Moderate"
            drug = np.random.choice(moderate_drugs)
        else:
            risk = "High"
            drug = np.random.choice(high_drugs)

        print(f"Risk Level: {risk}, Drug: {drug}")
        return render_template("result.html", risk=risk, drug=drug)

    except ValueError as ve:
        return f"Input Error: {str(ve)}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)