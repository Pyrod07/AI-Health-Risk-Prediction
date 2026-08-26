import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = "ml/dataset/heart.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 2. CHECK DATA
# ==========================================

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["target"].value_counts())


# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

X = df.drop("target", axis=1)
y = df["target"]


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 5. LOGISTIC REGRESSION
# ==========================================

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])


# ==========================================
# 6. RANDOM FOREST
# ==========================================

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# ==========================================
# 7. TRAIN MODELS
# ==========================================

print("\nTraining Logistic Regression...")
logistic_model.fit(X_train, y_train)

print("Training Random Forest...")
random_forest_model.fit(X_train, y_train)


# ==========================================
# 8. EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, model):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\n======================================")
    print(name)
    print("======================================")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return roc_auc


# ==========================================
# 9. EVALUATE MODELS
# ==========================================

logistic_auc = evaluate_model(
    "Logistic Regression",
    logistic_model
)

random_forest_auc = evaluate_model(
    "Random Forest",
    random_forest_model
)


# ==========================================
# 10. SELECT BEST MODEL
# ==========================================

if random_forest_auc >= logistic_auc:

    best_model = random_forest_model
    best_model_name = "Random Forest"

else:

    best_model = logistic_model
    best_model_name = "Logistic Regression"


print("\n======================================")
print("BEST MODEL")
print("======================================")

print("Selected:", best_model_name)


# ==========================================
# 11. SAVE MODEL
# ==========================================

MODEL_PATH = "ml/model/heart_disease_model.pkl"

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)