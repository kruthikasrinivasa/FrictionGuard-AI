# ============================================================
# FRICTIONGUARD AI - FRAUD DETECTION MODEL TRAINING
# ============================================================
# Final training design:
#   1. Chronological FIT period
#   2. Chronological VALIDATION period for threshold/model choice
#   3. Untouched chronological TEST period for final metrics
#
# No random train/validation split is used.
# The final model is refit on the complete training period
# (FIT + VALIDATION) after the threshold is selected.
# ============================================================

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "engineered_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

CHUNK_SIZE = 100_000
RANDOM_STATE = 42

# Chronological split of the complete engineered dataset:
# 64% -> model fit
# 16% -> validation / threshold selection
# 20% -> final untouched test
FIT_RATIO = 0.64
VALIDATION_RATIO = 0.16

# Keep all fraud transactions in every period.
# Cap legitimate transactions to keep training practical.
MAX_NORMAL_FIT = 300_000
MAX_NORMAL_VALIDATION = 100_000
MAX_NORMAL_TEST = 100_000

# False-positive budget used for threshold selection.
# We try the strictest budget first.
FPR_BUDGETS = (0.01, 0.02, 0.05, 0.10)

TARGET = "isFraud"

CATEGORICAL_FEATURES = [
    "type",
]

NUMERICAL_FEATURES = [
    "step",
    "amount",

    "sender_txn_count_before",
    "sender_avg_amount_before",

    "receiver_txn_count_before",
    "receiver_avg_amount_before",

    "sender_amount_vs_avg",
    "receiver_amount_vs_avg",

    "sender_txn_count_recent",
    "receiver_txn_count_recent",

    "sender_amount_recent",
    "receiver_amount_recent",

    "sender_old_balance",
    "receiver_old_balance",

    "sender_balance_to_amount",
    "receiver_balance_to_amount",

    "sender_zero_balance",
    "receiver_zero_balance",
]

ALL_FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERICAL_FEATURES
)


# ============================================================
# 3. DISPLAY HEADER
# ============================================================

print("=" * 75)
print("FRICTIONGUARD AI - FRAUD DETECTION MODEL TRAINING")
print("=" * 75)

print(f"Dataset: {DATA_PATH}")
print(f"Dataset exists: {DATA_PATH.exists()}")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Engineered dataset not found:\n{DATA_PATH}"
    )


# ============================================================
# 4. VERIFY COLUMNS
# ============================================================

print("\nChecking dataset columns...")

sample = pd.read_csv(
    DATA_PATH,
    nrows=5
)

missing_features = [
    col
    for col in ALL_FEATURES + [TARGET]
    if col not in sample.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )

print(f"Total columns found: {len(sample.columns)}")
print("Required features verified successfully.")


# ============================================================
# 5. COUNT TOTAL ROWS
# ============================================================

print("\n" + "=" * 75)
print("BUILDING CHRONOLOGICAL FIT / VALIDATION / TEST DATA")
print("=" * 75)

print("Using chunked loading to control memory usage.")

with open(DATA_PATH, "rb") as f:
    total_rows = sum(1 for _ in f) - 1

print(f"Total rows: {total_rows:,}")

FIT_END_ROW = int(total_rows * FIT_RATIO)

VALIDATION_END_ROW = int(
    total_rows * (FIT_RATIO + VALIDATION_RATIO)
)

print(
    f"Fit period ends around row: "
    f"{FIT_END_ROW:,}"
)

print(
    f"Validation period ends around row: "
    f"{VALIDATION_END_ROW:,}"
)

print(
    f"Test period starts around row: "
    f"{VALIDATION_END_ROW:,}"
)


# ============================================================
# 6. STORAGE FOR EACH CHRONOLOGICAL PERIOD
# ============================================================

fit_normal_parts = []
fit_fraud_parts = []

validation_normal_parts = []
validation_fraud_parts = []

test_normal_parts = []
test_fraud_parts = []

fit_normal_count = 0
validation_normal_count = 0
test_normal_count = 0

processed_rows = 0


# ============================================================
# 7. READ DATA CHRONOLOGICALLY
# ============================================================

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    start_row = processed_rows
    end_row = processed_rows + len(chunk)

    processed_rows = end_row

    print(
        f"\rReading chunk {chunk_number} | "
        f"Rows processed: "
        f"{processed_rows:,}/{total_rows:,}",
        end="",
        flush=True
    )

    chunk = chunk.dropna(
        subset=[TARGET]
    )

    # --------------------------------------------------------
    # FIT PERIOD
    # --------------------------------------------------------

    if start_row < FIT_END_ROW:

        fraud = chunk[
            chunk[TARGET] == 1
        ]

        if len(fraud) > 0:
            fit_fraud_parts.append(
                fraud[
                    ALL_FEATURES + [TARGET]
                ].copy()
            )

        normal = chunk[
            chunk[TARGET] == 0
        ]

        remaining = (
            MAX_NORMAL_FIT
            - fit_normal_count
        )

        if remaining > 0 and len(normal) > 0:

            take = min(
                remaining,
                len(normal)
            )

            # Preserve chronological order by taking
            # the earliest eligible legitimate rows.
            normal = normal.iloc[
                :take
            ]

            fit_normal_parts.append(
                normal[
                    ALL_FEATURES + [TARGET]
                ].copy()
            )

            fit_normal_count += len(normal)

    # --------------------------------------------------------
    # VALIDATION PERIOD
    # --------------------------------------------------------

    elif start_row < VALIDATION_END_ROW:

        fraud = chunk[
            chunk[TARGET] == 1
        ]

        if len(fraud) > 0:
            validation_fraud_parts.append(
                fraud[
                    ALL_FEATURES + [TARGET]
                ].copy()
            )

        normal = chunk[
            chunk[TARGET] == 0
        ]

        remaining = (
            MAX_NORMAL_VALIDATION
            - validation_normal_count
        )

        if remaining > 0 and len(normal) > 0:

            take = min(
                remaining,
                len(normal)
            )

            normal = normal.iloc[
                :take
            ]

            validation_normal_parts.append(
                normal[
                    ALL_FEATURES + [TARGET]
                ].copy()
            )

            validation_normal_count += len(normal)

    # --------------------------------------------------------
    # TEST PERIOD
    # --------------------------------------------------------

    else:

        fraud = chunk[
            chunk[TARGET] == 1
        ]

        if len(fraud) > 0:
            test_fraud_parts.append(
                fraud[
                    ALL_FEATURES + [TARGET]
                ].copy()
            )

        normal = chunk[
            chunk[TARGET] == 0
        ]

        remaining = (
            MAX_NORMAL_TEST
            - test_normal_count
        )

        if remaining > 0 and len(normal) > 0:

            take = min(
                remaining,
                len(normal)
            )

            normal = normal.iloc[
                :take
            ]

            test_normal_parts.append(
                normal[
                    ALL_FEATURES + [TARGET]
                ].copy()
            )

            test_normal_count += len(normal)


print("\n\nData extraction completed.")


# ============================================================
# 8. COMBINE EACH PERIOD
# ============================================================

print("\nCombining chronological periods...")

fit_normal = pd.concat(
    fit_normal_parts,
    ignore_index=True
)

fit_fraud = pd.concat(
    fit_fraud_parts,
    ignore_index=True
)

validation_normal = pd.concat(
    validation_normal_parts,
    ignore_index=True
)

if validation_fraud_parts:
    validation_fraud = pd.concat(
        validation_fraud_parts,
        ignore_index=True
    )
else:
    validation_fraud = pd.DataFrame(
        columns=ALL_FEATURES + [TARGET]
    )

test_normal = pd.concat(
    test_normal_parts,
    ignore_index=True
)

if test_fraud_parts:
    test_fraud = pd.concat(
        test_fraud_parts,
        ignore_index=True
    )
else:
    test_fraud = pd.DataFrame(
        columns=ALL_FEATURES + [TARGET]
    )


fit_df = pd.concat(
    [fit_normal, fit_fraud],
    ignore_index=True
)

validation_df = pd.concat(
    [validation_normal, validation_fraud],
    ignore_index=True
)

test_df = pd.concat(
    [test_normal, test_fraud],
    ignore_index=True
)


# ============================================================
# 9. SHUFFLE WITHIN EACH PERIOD ONLY
# ============================================================
# Shuffling within a period does not mix future data into
# earlier periods. It simply prevents training order effects.

fit_df = fit_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

validation_df = validation_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

test_df = test_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ============================================================
# 10. COMBINE FIT + VALIDATION FOR FINAL TRAINING LATER
# ============================================================

training_df = pd.concat(
    [fit_df, validation_df],
    ignore_index=True
)


# ============================================================
# 11. DATASET SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("CHRONOLOGICAL DATA SUMMARY")
print("=" * 75)

print("\nFIT PERIOD")
print(f"Rows  : {len(fit_df):,}")
print(
    f"Fraud : "
    f"{int(fit_df[TARGET].sum()):,}"
)
print(
    f"Normal: "
    f"{int((fit_df[TARGET] == 0).sum()):,}"
)

print("\nVALIDATION PERIOD")
print(f"Rows  : {len(validation_df):,}")
print(
    f"Fraud : "
    f"{int(validation_df[TARGET].sum()):,}"
)
print(
    f"Normal: "
    f"{int((validation_df[TARGET] == 0).sum()):,}"
)

print("\nFINAL TEST PERIOD")
print(f"Rows  : {len(test_df):,}")
print(
    f"Fraud : "
    f"{int(test_df[TARGET].sum()):,}"
)
print(
    f"Normal: "
    f"{int((test_df[TARGET] == 0).sum()):,}"
)

print("\nFINAL TRAINING PERIOD (FIT + VALIDATION)")
print(f"Rows  : {len(training_df):,}")
print(
    f"Fraud : "
    f"{int(training_df[TARGET].sum()):,}"
)
print(
    f"Normal: "
    f"{int((training_df[TARGET] == 0).sum()):,}"
)


# ============================================================
# 12. CREATE X / y
# ============================================================

X_fit = fit_df[ALL_FEATURES]
y_fit = fit_df[TARGET].astype(int)

X_validation = validation_df[ALL_FEATURES]
y_validation = validation_df[TARGET].astype(int)

X_training = training_df[ALL_FEATURES]
y_training = training_df[TARGET].astype(int)

X_test = test_df[ALL_FEATURES]
y_test = test_df[TARGET].astype(int)


# ============================================================
# 13. PREPROCESSING PIPELINE
# ============================================================

print("\nPreparing preprocessing pipeline...")

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            NUMERICAL_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        ),
    ]
)


# ============================================================
# 14. MODEL BUILDERS
# ============================================================

def build_logistic_model():

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=500,
                    solver="liblinear",
                    random_state=RANDOM_STATE
                )
            ),
        ]
    )


def build_random_forest_model():

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=18,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE
                )
            ),
        ]
    )


# ============================================================
# 15. THRESHOLD SELECTION
# ============================================================

def choose_threshold(
    model,
    X_validation,
    y_validation
):

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    candidates = []

    for threshold in np.arange(
        0.01,
        0.991,
        0.01
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0
        )

        cm = confusion_matrix(
            y_validation,
            predictions
        )

        false_positives = int(cm[0, 1])
        true_negatives = int(cm[0, 0])

        fpr = (
            false_positives
            / max(
                false_positives + true_negatives,
                1
            )
        )

        candidates.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "fpr": float(fpr),
                "false_positives": false_positives,
            }
        )

    # First try a strict 1% FPR budget.
    # Relax only when no threshold satisfies it.
    for budget in FPR_BUDGETS:

        eligible = [
            item
            for item in candidates
            if item["fpr"] <= budget
        ]

        if eligible:

            # Within the allowed FPR budget:
            # 1. maximize recall
            # 2. maximize precision
            # 3. maximize F1
            # 4. prefer the higher threshold
            chosen = max(
                eligible,
                key=lambda item: (
                    item["recall"],
                    item["precision"],
                    item["f1"],
                    item["threshold"],
                )
            )

            return chosen, budget

    # Defensive fallback.
    chosen = max(
        candidates,
        key=lambda item: (
            item["f1"],
            item["precision"],
            item["recall"],
        )
    )

    return chosen, None


# ============================================================
# 16. VALIDATION METRICS
# ============================================================

def validation_summary(
    model,
    X_validation,
    y_validation,
    model_name
):

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    pr_auc = average_precision_score(
        y_validation,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_validation,
        probabilities
    )

    threshold_info, budget = choose_threshold(
        model,
        X_validation,
        y_validation
    )

    print("\n" + "-" * 75)
    print(
        f"{model_name} - VALIDATION"
    )
    print("-" * 75)

    print(
        f"PR-AUC              : "
        f"{pr_auc:.4f}"
    )

    print(
        f"ROC-AUC             : "
        f"{roc_auc:.4f}"
    )

    print(
        f"Chosen threshold    : "
        f"{threshold_info['threshold']:.2f}"
    )

    print(
        f"Validation precision: "
        f"{threshold_info['precision']:.4f}"
    )

    print(
        f"Validation recall   : "
        f"{threshold_info['recall']:.4f}"
    )

    print(
        f"Validation F1       : "
        f"{threshold_info['f1']:.4f}"
    )

    print(
        f"Validation FPR      : "
        f"{threshold_info['fpr']:.4%}"
    )

    print(
        f"Validation FP       : "
        f"{threshold_info['false_positives']:,}"
    )

    if budget is not None:
        print(
            f"FPR budget used     : "
            f"{budget:.0%}"
        )
    else:
        print(
            "FPR budget used     : fallback"
        )

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "threshold": threshold_info["threshold"],
        "precision": threshold_info["precision"],
        "recall": threshold_info["recall"],
        "f1": threshold_info["f1"],
        "fpr": threshold_info["fpr"],
        "false_positives": threshold_info["false_positives"],
    }


# ============================================================
# 17. TRAIN LOGISTIC REGRESSION ON FIT PERIOD
# ============================================================

print("\n" + "=" * 75)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 75)

logistic_model = build_logistic_model()

print(
    "Training Logistic Regression "
    "on chronological fit period..."
)

logistic_model.fit(
    X_fit,
    y_fit
)

print(
    "Logistic Regression training completed."
)

logistic_validation = validation_summary(
    logistic_model,
    X_validation,
    y_validation,
    "LOGISTIC REGRESSION"
)


# ============================================================
# 18. TRAIN RANDOM FOREST ON FIT PERIOD
# ============================================================

print("\n" + "=" * 75)
print("MODEL 2: RANDOM FOREST")
print("=" * 75)

print(
    "Training Random Forest. "
    "This may take several minutes..."
)

random_forest_model = build_random_forest_model()

random_forest_model.fit(
    X_fit,
    y_fit
)

print(
    "Random Forest training completed."
)

rf_validation = validation_summary(
    random_forest_model,
    X_validation,
    y_validation,
    "RANDOM FOREST"
)


# ============================================================
# 19. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("VALIDATION MODEL COMPARISON")
print("=" * 75)

comparison = pd.DataFrame(
    [
        {
            "Model": "Logistic Regression",
            **logistic_validation
        },
        {
            "Model": "Random Forest",
            **rf_validation
        },
    ]
)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 20. SELECT BEST MODEL
# ============================================================
# PR-AUC is the primary ranking metric because the fraud class
# is highly imbalanced. The threshold is selected independently
# on validation data.

if (
    rf_validation["pr_auc"]
    >= logistic_validation["pr_auc"]
):

    best_name = "Random Forest"
    selected_threshold = (
        rf_validation["threshold"]
    )

else:

    best_name = "Logistic Regression"
    selected_threshold = (
        logistic_validation["threshold"]
    )


print("\n" + "=" * 75)
print("BEST MODEL")
print("=" * 75)

print(
    f"Selected model       : "
    f"{best_name}"
)

print(
    f"Selected threshold   : "
    f"{selected_threshold:.2f}"
)

print(
    "Selection criterion  : "
    "highest validation PR-AUC"
)


# ============================================================
# 21. REFIT SELECTED MODEL ON FULL TRAINING PERIOD
# ============================================================
# Validation is now finished and must not be used to tune anything
# else. The selected model is retrained on FIT + VALIDATION.
# The TEST period remains untouched.

print("\n" + "=" * 75)
print("REFITTING SELECTED MODEL ON FULL TRAINING PERIOD")
print("=" * 75)

if best_name == "Random Forest":
    best_model = build_random_forest_model()
else:
    best_model = build_logistic_model()

best_model.fit(
    X_training,
    y_training
)

print(
    "Full training-period refit completed."
)


# ============================================================
# 22. FINAL TEST EVALUATION
# ============================================================

def evaluate_final_model(
    model,
    X_test,
    y_test,
    threshold,
    model_name
):

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

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

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    true_negatives = int(cm[0, 0])
    false_positives = int(cm[0, 1])
    false_negatives = int(cm[1, 0])
    true_positives = int(cm[1, 1])

    false_positive_rate = (
        false_positives
        / max(
            true_negatives + false_positives,
            1
        )
    )

    fraud_detection_rate = (
        true_positives
        / max(
            true_positives + false_negatives,
            1
        )
    )

    print("\n" + "=" * 75)
    print(
        f"{model_name} - FINAL TEST EVALUATION"
    )
    print("=" * 75)

    print(
        f"Decision threshold : "
        f"{threshold:.4f}"
    )

    print(
        f"Precision          : "
        f"{precision:.4f}"
    )

    print(
        f"Recall             : "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score           : "
        f"{f1:.4f}"
    )

    print(
        f"ROC-AUC            : "
        f"{roc_auc:.4f}"
    )

    print(
        f"PR-AUC             : "
        f"{pr_auc:.4f}"
    )

    print(
        f"True positives     : "
        f"{true_positives:,}"
    )

    print(
        f"False positives    : "
        f"{false_positives:,}"
    )

    print(
        f"True negatives     : "
        f"{true_negatives:,}"
    )

    print(
        f"False negatives    : "
        f"{false_negatives:,}"
    )

    print(
        f"False positive rate: "
        f"{false_positive_rate:.4%}"
    )

    print(
        f"Fraud detection rate: "
        f"{fraud_detection_rate:.4%}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(cm)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "false_positive_rate": false_positive_rate,
        "fraud_detection_rate": fraud_detection_rate,
    }


final_results = evaluate_final_model(
    best_model,
    X_test,
    y_test,
    selected_threshold,
    best_name
)


# ============================================================
# 23. SAVE MODEL
# ============================================================

MODEL_PATH = (
    MODEL_DIR
    / "frictionguard_fraud_model.joblib"
)

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nModel saved successfully:")
print(MODEL_PATH)


# ============================================================
# 24. SAVE DECISION THRESHOLD
# ============================================================

THRESHOLD_PATH = (
    MODEL_DIR
    / "fraud_threshold.txt"
)

THRESHOLD_PATH.write_text(
    f"{selected_threshold:.4f}",
    encoding="utf-8"
)

print("\nDecision threshold saved:")
print(THRESHOLD_PATH)


# ============================================================
# 25. SAVE FINAL RESULTS
# ============================================================

RESULTS_PATH = (
    MODEL_DIR
    / "model_results.csv"
)

final_results_row = {
    "Model": best_name,
    **final_results
}

pd.DataFrame(
    [final_results_row]
).to_csv(
    RESULTS_PATH,
    index=False
)

print("\nModel results saved:")
print(RESULTS_PATH)


# ============================================================
# 26. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FRICTIONGUARD AI - MODEL TRAINING COMPLETE")
print("=" * 75)

print(
    f"Best model          : "
    f"{best_name}"
)

print(
    f"Fit samples         : "
    f"{len(fit_df):,}"
)

print(
    f"Validation samples  : "
    f"{len(validation_df):,}"
)

print(
    f"Training samples    : "
    f"{len(training_df):,}"
)

print(
    f"Testing samples     : "
    f"{len(test_df):,}"
)

print(
    f"Fraud test cases    : "
    f"{int(y_test.sum()):,}"
)

print("\nFINAL TEST METRICS")

print(
    f"Threshold           : "
    f"{final_results['threshold']:.4f}"
)

print(
    f"Precision           : "
    f"{final_results['precision']:.4f}"
)

print(
    f"Recall              : "
    f"{final_results['recall']:.4f}"
)

print(
    f"F1 Score            : "
    f"{final_results['f1']:.4f}"
)

print(
    f"PR-AUC              : "
    f"{final_results['pr_auc']:.4f}"
)

print(
    f"ROC-AUC             : "
    f"{final_results['roc_auc']:.4f}"
)

print(
    f"False positives     : "
    f"{final_results['false_positives']:,}"
)

print(
    f"False positive rate : "
    f"{final_results['false_positive_rate']:.4%}"
)

print(
    f"Fraud detection rate: "
    f"{final_results['fraud_detection_rate']:.4%}"
)

print("\nModel file:")
print(MODEL_PATH)

print("\n" + "=" * 75)
print("READY FOR FRAUD RISK ENGINE")
print("=" * 75)
