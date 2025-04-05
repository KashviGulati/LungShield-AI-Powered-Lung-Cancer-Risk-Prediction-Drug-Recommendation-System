import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the dataset
df = pd.read_csv("d:/Cancer risk prediction/dataset/Integrated_Lung_Cancer_Data.csv")
df.columns = df.columns.str.strip()

# Check original class balance
print("Original Class Distribution:\n", df["LUNG_CANCER"].value_counts(normalize=True))

# Balance the dataset
n_minority = df['LUNG_CANCER'].value_counts()[0]  # Number of Low Risk samples
df_0 = df[df['LUNG_CANCER'] == 0]
df_1 = df[df['LUNG_CANCER'] == 1].sample(n=n_minority, random_state=42)  # Undersample High Risk
df_balanced = pd.concat([df_0, df_1])
print("Balanced Class Distribution:\n", df_balanced["LUNG_CANCER"].value_counts(normalize=True))

# Analyze correlation and specific cases
print("\nCorrelation with LUNG_CANCER (Balanced Data):")
print(df_balanced[['AGE', 'GENDER', 'SMOKING', 'ANXIETY', 'FATIGUE', 'COUGHING', 'LUNG_CANCER']].corr()['LUNG_CANCER'])
high_anxiety_fatigue = df_balanced[(df_balanced['ANXIETY'] == 2) & (df_balanced['FATIGUE'] == 2)]
print("\nCases with ANXIETY=2 and FATIGUE=2 (Balanced Data):")
print(high_anxiety_fatigue['LUNG_CANCER'].value_counts(normalize=True))

# Define features
selected_features = ['AGE', 'GENDER', 'SMOKING', 'ANXIETY', 'FATIGUE', 'COUGHING']
X = df_balanced[selected_features]
y = df_balanced["LUNG_CANCER"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model with limited depth
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Random Forest Accuracy: {acc:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"✅ Cross-validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Feature importance
print("Feature Importances:", dict(zip(selected_features, model.feature_importances_)))

# Save the model and scaler
with open("d:/Cancer risk prediction/cancer_risk_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("d:/Cancer risk prediction/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully.")