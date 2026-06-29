#data_preprocessing.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Path to raw dataset folder
RAW_DATA_PATH = os.path.join("dataset", "raw")

# Get all CSV files
files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith(".csv")]

print("Files Found:", files)

# Load and merge all datasets
dataframes = []

for file in files:
    file_path = os.path.join(RAW_DATA_PATH, file)
    print(f"Loading {file}...")
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Merge all files
data = pd.concat(dataframes, ignore_index=True)

print("\nMerged Dataset Shape:", data.shape)

# Remove duplicates
data.drop_duplicates(inplace=True)

# Replace infinite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop missing values
data.dropna(inplace=True)

print("Shape After Cleaning:", data.shape)

# Fix column names (remove spaces)
data.columns = data.columns.str.strip()

# Convert label to binary
data['Label'] = data['Label'].apply(lambda x: 0 if x == "BENIGN" else 1)

print("\nClass Distribution Before SMOTE:")
print(data['Label'].value_counts())

# Separate features and target
X = data.drop("Label", axis=1)
y = data["Label"]

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE ONLY on training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nClass Distribution After SMOTE (Training Set):")
print(pd.Series(y_train_balanced).value_counts())

# Save balanced train and original test
train_data = pd.concat([pd.DataFrame(X_train_balanced), pd.Series(y_train_balanced, name="Label")], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv(os.path.join("dataset", "train.csv"), index=False)
test_data.to_csv(os.path.join("dataset", "test.csv"), index=False)

print("\nBalanced Train & Test files saved successfully!")