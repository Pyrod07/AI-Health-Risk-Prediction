# ❤️ AI Health Risk Prediction

An AI-powered web application that predicts cardiovascular disease risk based on patient health parameters using machine learning.

The project combines a **Random Forest machine learning model**, **FastAPI backend**, **SQLite database**, and a **responsive HTML/CSS/JavaScript frontend**.

---

## 🚀 Features

- ❤️ Cardiovascular disease risk prediction
- 🤖 Machine learning based prediction
- 📊 Risk probability percentage
- 🟢 Low / 🟡 Moderate / 🔴 High risk classification
- 🧠 Random Forest classification model
- ⚡ FastAPI REST API
- 💾 SQLite prediction history
- 🌐 Interactive web interface
- 📈 Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
- 📊 Feature importance analysis
- 🔄 Prediction history

---

## 🏗️ Project Architecture

```text
AI-Health-Risk-Prediction/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── database.py
│   │   ├── models.py
│   │   ├── schemas.py
│   │   └── ml/
│   │       └── predictor.py
│   │
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   └── config.js
│
├── ml/
│   ├── dataset/
│   │   └── heart.csv
│   ├── model/
│   │   └── heart_disease_model.pkl
│   └── train.py
│
├── .gitignore
└── README.md