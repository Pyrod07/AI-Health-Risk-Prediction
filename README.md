\# 🫀 AI Health Risk Prediction System



An end-to-end \*\*AI-powered cardiovascular health risk prediction system\*\* that uses machine learning to estimate heart disease risk from 13 clinical parameters.



The project combines a \*\*Random Forest machine learning model\*\*, \*\*FastAPI backend\*\*, \*\*SQLite database\*\*, and a responsive \*\*HTML/CSS/JavaScript frontend\*\*.



> ⚠️ \*\*Disclaimer:\*\* This project is intended for educational and demonstration purposes only. It is not a medical diagnostic system and should not be used to make clinical decisions.



\---



\## 🚀 Features



\- 🧠 Machine learning-based cardiovascular risk prediction

\- 🌲 Random Forest classification model

\- 📊 Risk probability and risk-level estimation

\- 🔬 13 clinical input parameters

\- ⚡ FastAPI REST API

\- 🗄️ SQLite database for prediction history

\- 📜 Prediction history tracking

\- 🌐 Responsive web frontend

\- 🔐 CORS-enabled frontend/backend communication

\- 📈 Model evaluation using multiple ML metrics

\- 🔄 Train and replace the ML model using the included training script

\- 📖 Interactive FastAPI Swagger documentation



\---



\## 🏗️ System Architecture



```text

&#x20;                   ┌──────────────────────┐

&#x20;                   │      Frontend        │

&#x20;                   │ HTML / CSS / JS      │

&#x20;                   └──────────┬───────────┘

&#x20;                              │

&#x20;                              │ HTTP / REST

&#x20;                              ▼

&#x20;                   ┌──────────────────────┐

&#x20;                   │      FastAPI         │

&#x20;                   │      Backend         │

&#x20;                   └──────────┬───────────┘

&#x20;                              │

&#x20;                   ┌──────────┴───────────┐

&#x20;                   │                      │

&#x20;                   ▼                      ▼

&#x20;         ┌──────────────────┐   ┌──────────────────┐

&#x20;         │ Random Forest ML │   │ SQLite Database  │

&#x20;         │      Model       │   │ Prediction       │

&#x20;         └──────────────────┘   │ History          │

&#x20;                                └──────────────────┘

