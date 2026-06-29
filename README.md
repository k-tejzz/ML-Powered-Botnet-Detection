# ML-Powered Botnet Detection Using Network Flow Analytics

## 📌 Overview

This project presents a Machine Learning-based Intrusion Detection System (IDS) for detecting botnet and malicious network traffic using **network flow analytics**. The system is trained on the **CICIDS2017** dataset and uses a **Random Forest Classifier** to classify network traffic as either **Benign** or **Attack**.

The project performs complete data preprocessing, class imbalance handling using SMOTE, model training, evaluation, and visualization of results.

---

## 🚀 Features

- Merge multiple CICIDS2017 CSV files
- Data preprocessing and cleaning
- Handle missing and infinite values
- Binary classification (Benign / Attack)
- Handle class imbalance using SMOTE
- Train Random Forest classifier
- Evaluate model using multiple performance metrics
- Generate visualization plots automatically
- Save trained model for future predictions

---

## 📂 Project Structure

```
Botnet-Detection-ML/
│
├── models/
│   └── random_forest_model.pkl
│
├── outputs/
│   ├── anomaly_score_distribution.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── roc_curve.png
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

Dataset Used:

**CICIDS2017**

The dataset is publicly available from the Canadian Institute for Cybersecurity (CIC).

> Due to GitHub file size limitations, the dataset is **not included** in this repository.

After downloading the dataset, place all CSV files inside:

```
dataset/raw/
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/k-tejzz/ML-Powered-Botnet-Detection.git
```

Go into the project

```bash
cd ML-Powered-Botnet-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Preprocess Dataset

```bash
python src/data_preprocessing.py
```

This will:

- Merge CSV files
- Clean the dataset
- Apply SMOTE
- Generate train.csv and test.csv

---

### 2. Train Model

```bash
python src/train.py
```

This saves the trained model inside:

```
models/random_forest_model.pkl
```

---

### 3. Evaluate Model

```bash
python src/evaluate.py
```

Evaluation automatically generates:

- Confusion Matrix
- ROC Curve
- Feature Importance
- Anomaly Score Distribution

inside the **outputs/** folder.

---

## 📈 Results

| Metric | Score |
|---------|-------|
| Accuracy | **99.87%** |
| Precision | **99.54%** |
| Recall | **99.67%** |
| F1 Score | **99.61%** |
| ROC-AUC | **99.97%** |

---

## 📊 Generated Plots

The evaluation script automatically saves the following visualizations:

- Confusion Matrix
- ROC Curve
- Feature Importance
- Anomaly Score Distribution

These are stored inside:

```
outputs/
```

---

## 🧠 Machine Learning Model

Algorithm Used:

**Random Forest Classifier**

### Why Random Forest?

- High accuracy on structured data
- Handles large datasets efficiently
- Resistant to overfitting
- Provides feature importance
- Fast prediction

---

## ⚖️ Handling Class Imbalance

The original dataset contained significantly more benign traffic than attack traffic.

To address this, **SMOTE (Synthetic Minority Oversampling Technique)** was applied **only to the training dataset**, ensuring balanced learning while preventing data leakage.

---

## 🔮 Future Improvements

- Compare with XGBoost and LightGBM
- Real-time network traffic monitoring
- Deep Learning based IDS
- Web Dashboard
- Live packet capture integration

---

## 👨‍💻 Author

**Satya Teja**

GitHub:
https://github.com/k-tejzz

---

## 📜 License

This project is intended for educational and research purposes.