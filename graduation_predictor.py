"""
Student Graduation Predictor
Trains a classification model on historical student data (2018–2025)
and predicts whether a new student will graduate within 4 years.
"""

import os
import glob
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATA_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(DATA_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(DATA_DIR, "encoders.pkl")

CATEGORICAL_COLS = ["sex", "department", "major", "minor", "location", "commute"]
NUMERICAL_COLS   = ["gpa", "high_school_score"]
FEATURES         = CATEGORICAL_COLS + NUMERICAL_COLS
TARGET           = "graduated_in_4_years"


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(DATA_DIR, "student_data_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No student_data_*.csv files found in {DATA_DIR}")

    frames = [pd.read_csv(f) for f in sorted(csv_files)]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} records from {len(csv_files)} files.")
    return df


# ─────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Encode categorical columns.
    If fit=True, creates and fits new encoders (training).
    If fit=False, uses provided encoders (prediction).
    Returns (X, y, encoders).
    """
    df = df.copy()

    # Encode target
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

    if fit:
        encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in CATEGORICAL_COLS:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda val, le=le: le.transform([val])[0]
                if val in le.classes_
                else -1       # unseen category → -1
            )

    X = df[FEATURES]
    y = df[TARGET]
    return X, y, encoders


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
def train():
    print("\n" + "=" * 55)
    print("  TRAINING STUDENT GRADUATION PREDICTOR")
    print("=" * 55)

    df = load_data()

    # Check class balance
    grad_pct = df[TARGET].value_counts(normalize=True) * 100
    print(f"\nTarget distribution:")
    for label, pct in grad_pct.items():
        print(f"  {label:<4}  {pct:.1f}%")

    X, y, encoders = preprocess(df, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── compare three models ──────────────────
    models = {
        "Logistic Regression":   LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":         RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":     GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    print("\n── Model Comparison ──────────────────────────")
    best_model, best_acc = None, 0
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"  {name:<25}  Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc, best_model, best_name = acc, model, name

    print(f"\nBest model → {best_name}  (accuracy {best_acc:.4f})")

    # ── detailed report for best model ───────
    y_pred = best_model.predict(X_test)
    print("\n── Classification Report ─────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Not Grad (4yr)", "Graduated (4yr)"]))

    # ── feature importance (tree models) ─────
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print("── Feature Importance ────────────────────────")
        for feat, imp in importances.items():
            bar = "█" * int(imp * 50)
            print(f"  {feat:<20} {imp:.4f}  {bar}")

    # ── save ──────────────────────────────────
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(encoders,   ENCODERS_PATH)
    print(f"\nModel saved   → {MODEL_PATH}")
    print(f"Encoders saved → {ENCODERS_PATH}")

    # ── confusion matrix plot ─────────────────
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not 4yr", "Graduated"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    cm_path = os.path.join(DATA_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.show()
    print(f"Confusion matrix saved → {cm_path}")

    return best_model, encoders


# ─────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────
def predict(student: dict):
    """
    Predict graduation for a single student.

    Parameters
    ----------
    student : dict with keys:
        sex              : "Male" | "Female"
        department       : e.g. "Computers", "Engineering", "Business" …
        major            : e.g. "Computer Science", "Finance" …
        minor            : e.g. "Mathematics", "Statistics" …
        gpa              : float  0.0 – 4.0
        high_school_score: int    0 – 100
        location         : "Edinburg" | "Brownsville" | "Harlingen"
        commute          : "Yes" | "No"

    Returns
    -------
    dict: prediction, probability, confidence label
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
        raise FileNotFoundError("Model not found. Run train() first.")

    model    = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)

    # Build a single-row DataFrame
    row = pd.DataFrame([{col: student[col] for col in FEATURES}])

    # Encode categoricals
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        val = str(row[col].iloc[0])
        row[col] = le.transform([val])[0] if val in le.classes_ else -1

    proba      = model.predict_proba(row)[0]   # [P(No), P(Yes)]
    prediction = int(model.predict(row)[0])
    prob_yes   = round(float(proba[1]) * 100, 1)

    result = {
        "will_graduate_in_4_years": "Yes" if prediction == 1 else "No",
        "probability":              f"{prob_yes}%",
        "confidence":               "High" if prob_yes >= 70 or prob_yes <= 30
                                    else "Medium" if prob_yes >= 55 or prob_yes <= 45
                                    else "Low",
    }
    return result


def predict_interactive():
    """Walk the user through entering student details and print the prediction."""
    print("\n" + "=" * 55)
    print("  PREDICT: Will this student graduate in 4 years?")
    print("=" * 55)

    if not os.path.exists(ENCODERS_PATH):
        print("No trained model found. Training first...\n")
        train()

    encoders = joblib.load(ENCODERS_PATH)

    def ask(prompt, valid=None):
        while True:
            val = input(f"  {prompt}: ").strip()
            if valid and val not in valid:
                print(f"    Choose from: {', '.join(valid)}")
            else:
                return val

    sex        = ask("Sex (Male/Female)", ["Male", "Female"])
    dept_opts  = list(encoders["department"].classes_)
    department = ask(f"Department {dept_opts}", dept_opts)
    major_opts = list(encoders["major"].classes_)
    major      = ask(f"Major {major_opts}", major_opts)
    minor_opts = list(encoders["minor"].classes_)
    minor      = ask(f"Minor {minor_opts}", minor_opts)
    loc_opts   = list(encoders["location"].classes_)
    location   = ask(f"Location {loc_opts}", loc_opts)
    commute    = ask("Commutes? (Yes/No)", ["Yes", "No"])

    while True:
        try:
            gpa = float(input("  GPA (0.0 – 4.0): ").strip())
            if 0.0 <= gpa <= 4.0:
                break
            print("    GPA must be between 0.0 and 4.0")
        except ValueError:
            print("    Enter a number like 3.5")

    while True:
        try:
            hs = int(input("  High School Score (0 – 100): ").strip())
            if 0 <= hs <= 100:
                break
            print("    Score must be 0–100")
        except ValueError:
            print("    Enter a whole number like 85")

    student = {
        "sex": sex, "department": department, "major": major,
        "minor": minor, "location": location, "commute": commute,
        "gpa": gpa, "high_school_score": hs,
    }

    result = predict(student)

    print("\n── Prediction Result ─────────────────────────")
    print(f"  Graduate in 4 years? → {result['will_graduate_in_4_years']}")
    print(f"  Probability          → {result['probability']}")
    print(f"  Confidence           → {result['confidence']}")
    print("─" * 55)


# ─────────────────────────────────────────────
#  MAIN MENU
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════╗")
    print("║   Student Graduation Predictor           ║")
    print("╚══════════════════════════════════════════╝")
    print("\n  1. Train model on historical data")
    print("  2. Predict for a new student")
    print("  3. Train then predict")

    choice = input("\nChoose (1 / 2 / 3): ").strip()

    if choice == "1":
        train()
    elif choice == "2":
        predict_interactive()
    elif choice == "3":
        train()
        predict_interactive()
    else:
        print("Invalid choice. Run the script again.")
