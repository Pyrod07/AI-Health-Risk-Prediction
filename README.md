## Deployment

Vercel hosts the static frontend. The FastAPI backend must run separately on a
container host such as Render, Railway, Fly.io, or Google Cloud Run.

### Run the API with Docker

Build from the repository root so the trained model in `ml/model` is included:

```bash
docker build -t ai-health-risk-api .
docker run --rm -p 8000:8000 -e CORS_ORIGINS=https://your-vercel-domain.vercel.app ai-health-risk-api
```

For local development:

```bash
docker compose up --build
```

### Deploy the frontend to Vercel

Import the repository into Vercel and leave the project root as `/`. The included
`vercel.json` publishes `frontend/` as the static output. After deploying the
API, replace the URL in `frontend/config.js` with its public HTTPS URL, then
redeploy the frontend. Set the API's `CORS_ORIGINS` to the Vercel deployment
origin.
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