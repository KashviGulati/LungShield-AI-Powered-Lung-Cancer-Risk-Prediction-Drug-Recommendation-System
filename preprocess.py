import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv(r"D:\Cancer risk prediction\dataset\Integrated_Lung_Cancer_Data.csv")
df.columns = df.columns.str.strip()

# Check class balance
print("Class Distribution:\n", df["LUNG_CANCER"].value_counts(normalize=True))

# Define features
selected_features = ['AGE', 'GENDER', 'SMOKING', 'ANXIETY', 'FATIGUE', 'COUGHING']
X = df[selected_features]
y = df["LUNG_CANCER"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model with class weighting
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Random Forest Accuracy: {acc:.2f}")

cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"✅ Cross-validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Feature importance
importances = dict(zip(selected_features, model.feature_importances_))
print("Feature Importances:", importances)

# Save the model and scaler
with open("cancer_risk_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully.")

# After loading df
print("\nCorrelation with LUNG_CANCER:")
print(df[selected_features + ['LUNG_CANCER']].corr()['LUNG_CANCER'])

high_anxiety_fatigue = df[(df['ANXIETY'] == 2) & (df['FATIGUE'] == 2)]
print("\nCases with ANXIETY=2 and FATIGUE=2:")
print(high_anxiety_fatigue['LUNG_CANCER'].value_counts(normalize=True))