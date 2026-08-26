import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = "ml/dataset/heart.csv"
MODEL_PATH = "ml/model/heart_disease_model.pkl"
RESULTS_PATH = "ml/results"

os.makedirs("ml/model", exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_PATH)

print("\n======================================")
print("DATASET")
print("======================================")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())


# ==========================================
# 3. DATA VALIDATION
# ==========================================

print("\n======================================")
print("DATA VALIDATION")
print("======================================")

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["target"].value_counts())


if df.isnull().sum().sum() > 0:
    raise ValueError("Dataset contains missing values.")

if "target" not in df.columns:
    raise ValueError("Target column not found.")

print("\nDataset validation successful.")


# ==========================================
# 4. FEATURES AND TARGET
# ==========================================

X = df.drop("target", axis=1)
y = df["target"]

FEATURE_NAMES = X.columns.tolist()

print("\nFeatures:")
for feature in FEATURE_NAMES:
    print("-", feature)


# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n======================================")
print("DATA SPLIT")
print("======================================")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ==========================================
# 6. DEFINE MODELS
# ==========================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ==========================================
# 7. CROSS VALIDATION
# ==========================================

print("\n======================================")
print("CROSS VALIDATION")
print("======================================")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = {}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc"
    )

    cv_scores[name] = scores.mean()

    print(
        f"{name}: "
        f"{scores.mean():.4f} "
        f"(± {scores.std():.4f})"
    )


# ==========================================
# 8. TRAIN AND EVALUATE MODELS
# ==========================================

print("\n======================================")
print("MODEL EVALUATION")
print("======================================")

evaluation_results = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

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

    evaluation_results[name] = {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }

    print("\n--------------------------------------")
    print(name)
    print("--------------------------------------")

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


# ==========================================
# 9. SELECT BEST MODEL
# ==========================================

best_model_name = max(
    evaluation_results,
    key=lambda name: evaluation_results[name]["roc_auc"]
)

best_model = evaluation_results[
    best_model_name
]["model"]

print("\n======================================")
print("BEST MODEL")
print("======================================")

print("Selected:", best_model_name)

print(
    "Test ROC-AUC:",
    round(
        evaluation_results[best_model_name]["roc_auc"],
        4
    )
)

print(
    "Cross-validation ROC-AUC:",
    round(
        cv_scores[best_model_name],
        4
    )
)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_predictions
)

print("\n======================================")
print("CONFUSION MATRIX")
print("======================================")

print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    f"{RESULTS_PATH}/confusion_matrix.png"
)

plt.close()


# ==========================================
# 11. ROC CURVES
# ==========================================

plt.figure()

for name, result in evaluation_results.items():

    RocCurveDisplay.from_predictions(
        y_test,
        result["model"].predict_proba(X_test)[:, 1],
        name=name
    )

plt.title("ROC Curve Comparison")

plt.tight_layout()

plt.savefig(
    f"{RESULTS_PATH}/roc_curves.png"
)

plt.close()


# ==========================================
# 12. FEATURE IMPORTANCE
# ==========================================

print("\n======================================")
print("FEATURE IMPORTANCE")
print("======================================")

if hasattr(best_model, "feature_importances_"):

    importances = best_model.feature_importances_

elif hasattr(
    best_model,
    "named_steps"
):

    classifier = best_model.named_steps[
        "classifier"
    ]

    if hasattr(
        classifier,
        "coef_"
    ):

        importances = abs(
            classifier.coef_[0]
        )

    else:

        importances = None

else:

    importances = None


if importances is not None:

    feature_importance = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": importances
    })

    feature_importance = (
        feature_importance
        .sort_values(
            "importance",
            ascending=False
        )
    )

    print(feature_importance.to_string(
        index=False
    ))

    feature_importance.to_csv(
        f"{RESULTS_PATH}/feature_importance.csv",
        index=False
    )


# ==========================================
# 13. SAVE MODEL
# ==========================================

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\n======================================")
print("MODEL SAVED")
print("======================================")

print("Model:", best_model_name)
print("Location:", MODEL_PATH)

print("\nTraining completed successfully!")