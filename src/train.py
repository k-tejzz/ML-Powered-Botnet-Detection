#train.py
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load training data
train_path = os.path.join("dataset", "train.csv")
train_data = pd.read_csv(train_path)

print("Training Data Loaded:", train_data.shape)

# Split features and target
X_train = train_data.drop("Label", axis=1)
y_train = train_data["Label"]

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest Model...")
model.fit(X_train, y_train)

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Save model
model_path = os.path.join("models", "random_forest_model.pkl")
joblib.dump(model, model_path)

print("Model training complete!")
print("Model saved at:", model_path)