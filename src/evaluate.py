#evalute.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Load test data
test_path = os.path.join("dataset", "test.csv")
test_data = pd.read_csv(test_path)

print("Test Data Loaded:", test_data.shape)

X_test = test_data.drop("Label", axis=1)
y_test = test_data["Label"]

# Load model
model_path = os.path.join("models", "random_forest_model.pkl")
model = joblib.load(model_path)

print("Model Loaded Successfully!")

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nEvaluation Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)

# ==============================
# 1️⃣ Confusion Matrix
# ==============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# ==============================
# 2️⃣ ROC Curve
# ==============================
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label="ROC Curve (AUC = %0.4f)" % roc_auc)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/roc_curve.png")
plt.close()

# ==============================
# 3️⃣ Feature Importance
# ==============================
importances = model.feature_importances_
feature_names = X_test.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Top 20 Important Features")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

# ==============================
# 4️⃣ Anomaly Score Distribution
# ==============================
plt.figure()
sns.histplot(y_prob, bins=50, kde=True)
plt.title("Anomaly Score Distribution")
plt.xlabel("Predicted Probability (Attack)")
plt.tight_layout()
plt.savefig("outputs/anomaly_score_distribution.png")
plt.close()

print("\nAll plots saved successfully in 'outputs' folder!")