# ============================================================
# FRICTIONGUARD AI - RISK ENGINE
# ============================================================
#
# Converts ML predictions + behavioral signals into:
#
#   1. Fraud probability
#   2. Risk score (0-100)
#   3. Risk level
#   4. Explainable fraud reasons
#   5. Recommended action
#
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "frictionguard_fraud_model.joblib"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "engineered_features.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RISK_THRESHOLD_REVIEW = 40
RISK_THRESHOLD_HIGH = 65
RISK_THRESHOLD_CRITICAL = 85


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 75)
print("FRICTIONGUARD AI - RISK ENGINE")
print("=" * 75)

print(f"Model: {MODEL_PATH}")
print(f"Model exists: {MODEL_PATH.exists()}")

print(f"Dataset: {DATA_PATH}")
print(f"Dataset exists: {DATA_PATH.exists()}")


# ============================================================
# 4. LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}\n\n"
        "Run train_model.py first."
    )


print("\nLoading trained fraud detection model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# 5. REQUIRED FEATURES
# ============================================================

REQUIRED_FEATURES = [

    "step",
    "type",
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


# ============================================================
# 6. LOAD SAMPLE TRANSACTIONS
# ============================================================

print("\nLoading sample transactions...")

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )


# Read only a manageable sample.
sample_df = pd.read_csv(
    DATA_PATH,
    nrows=1000
)


missing = [
    column
    for column in REQUIRED_FEATURES
    if column not in sample_df.columns
]


if missing:

    raise ValueError(
        f"Missing features:\n{missing}"
    )


print(
    f"Loaded {len(sample_df):,} transactions."
)


# ============================================================
# 7. ML PREDICTION
# ============================================================

print("\nGenerating ML fraud probabilities...")

X = sample_df[REQUIRED_FEATURES]

fraud_probabilities = model.predict_proba(X)[:, 1]

sample_df["ml_fraud_probability"] = (
    fraud_probabilities
)


# ============================================================
# 8. HELPER FUNCTIONS
# ============================================================

def safe_value(row, column):

    value = row.get(column, 0)

    if pd.isna(value):
        return 0.0

    return float(value)


# ============================================================
# 9. BEHAVIORAL RISK SCORING
# ============================================================

def calculate_behavior_score(row):

    score = 0.0
    reasons = []


    # --------------------------------------------------------
    # Amount anomaly
    # --------------------------------------------------------

    sender_ratio = safe_value(
        row,
        "sender_amount_vs_avg"
    )

    receiver_ratio = safe_value(
        row,
        "receiver_amount_vs_avg"
    )


    # Very unusual sender amount
    if sender_ratio >= 10:

        score += 18

        reasons.append(
            "Transaction amount is extremely high "
            "compared with sender's historical average"
        )

    elif sender_ratio >= 5:

        score += 12

        reasons.append(
            "Transaction amount is significantly "
            "higher than sender's normal behavior"
        )

    elif sender_ratio >= 3:

        score += 7

        reasons.append(
            "Transaction amount is above sender's "
            "historical pattern"
        )


    # --------------------------------------------------------
    # Receiver anomaly
    # --------------------------------------------------------

    if receiver_ratio >= 10:

        score += 15

        reasons.append(
            "Amount is highly unusual for the receiver"
        )

    elif receiver_ratio >= 5:

        score += 10

        reasons.append(
            "Receiver is handling an unusually large amount"
        )


    # --------------------------------------------------------
    # Transaction frequency
    # --------------------------------------------------------

    sender_recent = safe_value(
        row,
        "sender_txn_count_recent"
    )

    receiver_recent = safe_value(
        row,
        "receiver_txn_count_recent"
    )


    if sender_recent >= 10:

        score += 12

        reasons.append(
            "High recent sender transaction activity"
        )

    elif sender_recent >= 5:

        score += 7

        reasons.append(
            "Elevated recent sender activity"
        )


    if receiver_recent >= 15:

        score += 10

        reasons.append(
            "Receiver shows unusually high recent activity"
        )

    elif receiver_recent >= 8:

        score += 5

        reasons.append(
            "Receiver has elevated recent activity"
        )


    # --------------------------------------------------------
    # Balance anomalies
    # --------------------------------------------------------

    sender_balance = safe_value(
        row,
        "sender_old_balance"
    )

    receiver_balance = safe_value(
        row,
        "receiver_old_balance"
    )

    amount = safe_value(
        row,
        "amount"
    )


    # Sender balance insufficient for transaction
    if amount > sender_balance and sender_balance > 0:

        score += 15

        reasons.append(
            "Transaction amount exceeds sender's "
            "available balance"
        )


    # Zero sender balance
    if (
        safe_value(
            row,
            "sender_zero_balance"
        ) == 1
    ):

        score += 8

        reasons.append(
            "Sender account has zero balance"
        )


    # Zero receiver balance
    if (
        safe_value(
            row,
            "receiver_zero_balance"
        ) == 1
    ):

        score += 5

        reasons.append(
            "Receiver account has zero balance"
        )


    # --------------------------------------------------------
    # Transaction type
    # --------------------------------------------------------

    transaction_type = str(
        row.get("type", "")
    ).upper()


    if transaction_type == "TRANSFER":

        score += 2

    elif transaction_type == "CASH_OUT":

        score += 3


    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return min(score, 70), reasons


# ============================================================
# 10. COMBINE ML + BEHAVIORAL SIGNALS
# ============================================================

def calculate_risk(row):

    ml_probability = safe_value(
        row,
        "ml_fraud_probability"
    )

    ml_score = ml_probability * 60


    behavioral_score, reasons = (
        calculate_behavior_score(row)
    )


    total_score = (
        ml_score
        + behavioral_score
    )


    total_score = min(
        max(total_score, 0),
        100
    )


    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if total_score >= RISK_THRESHOLD_CRITICAL:

        risk_level = "CRITICAL"
        action = "BLOCK / MANUAL INVESTIGATION"

    elif total_score >= RISK_THRESHOLD_HIGH:

        risk_level = "HIGH"
        action = "STEP-UP AUTHENTICATION / REVIEW"

    elif total_score >= RISK_THRESHOLD_REVIEW:

        risk_level = "MEDIUM"
        action = "ADDITIONAL VERIFICATION"

    else:

        risk_level = "LOW"
        action = "ALLOW"


    # --------------------------------------------------------
    # ML reason
    # --------------------------------------------------------

    if ml_probability >= 0.90:

        reasons.insert(
            0,
            "Machine-learning model assigns very high "
            "fraud probability"
        )

    elif ml_probability >= 0.70:

        reasons.insert(
            0,
            "Machine-learning model assigns elevated "
            "fraud probability"
        )

    elif ml_probability >= 0.40:

        reasons.insert(
            0,
            "Machine-learning model detects suspicious "
            "transaction characteristics"
        )


    # If no reasons
    if not reasons:

        reasons.append(
            "No significant anomalous behavior detected"
        )


    return {
        "risk_score": round(total_score, 2),
        "risk_level": risk_level,
        "recommended_action": action,
        "ml_probability": round(
            ml_probability * 100,
            2
        ),
        "behavior_score": round(
            behavioral_score,
            2
        ),
        "reasons": reasons[:5],
    }


# ============================================================
# 11. ANALYZE TRANSACTION
# ============================================================

def analyze_transaction(row):

    result = calculate_risk(row)

    return result


# ============================================================
# 12. RUN RISK ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("RUNNING FRICTIONGUARD RISK ENGINE")
print("=" * 75)


results = []


for index, row in sample_df.iterrows():

    result = analyze_transaction(row)

    results.append(result)


# ============================================================
# 13. ADD RESULTS TO DATAFRAME
# ============================================================

sample_df["risk_score"] = [
    result["risk_score"]
    for result in results
]

sample_df["risk_level"] = [
    result["risk_level"]
    for result in results
]

sample_df["recommended_action"] = [
    result["recommended_action"]
    for result in results
]

sample_df["behavior_score"] = [
    result["behavior_score"]
    for result in results
]

sample_df["ml_probability_percent"] = [
    result["ml_probability"]
    for result in results
]


# ============================================================
# 14. DISPLAY HIGH-RISK TRANSACTIONS
# ============================================================

print("\n" + "=" * 75)
print("TOP HIGH-RISK TRANSACTIONS")
print("=" * 75)


top_risk = sample_df.sort_values(
    "risk_score",
    ascending=False
).head(10)


display_columns = [
    "step",
    "type",
    "amount",
    "ml_probability_percent",
    "behavior_score",
    "risk_score",
    "risk_level",
    "recommended_action",
]


print(
    top_risk[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# 15. RISK DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("RISK DISTRIBUTION")
print("=" * 75)


risk_distribution = (
    sample_df["risk_level"]
    .value_counts()
)


for level in [
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
]:

    count = int(
        risk_distribution.get(
            level,
            0
        )
    )

    percentage = (
        count
        / len(sample_df)
        * 100
    )

    print(
        f"{level:10s}: "
        f"{count:5d} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# 16. EXPORT RISK RESULTS
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OUTPUT_PATH = (
    OUTPUT_DIR
    / "risk_scored_transactions.csv"
)


sample_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 17. SHOW EXPLAINABLE EXAMPLE
# ============================================================

print("\n" + "=" * 75)
print("EXPLAINABLE FRAUD DECISION")
print("=" * 75)


highest_risk = (
    top_risk.iloc[0]
)

highest_result = analyze_transaction(
    highest_risk
)


print(
    f"\nTransaction type : "
    f"{highest_risk['type']}"
)

print(
    f"Transaction amount : "
    f"{highest_risk['amount']:,.2f}"
)

print(
    f"\nML fraud probability : "
    f"{highest_result['ml_probability']:.2f}%"
)

print(
    f"Behavior score       : "
    f"{highest_result['behavior_score']:.2f}/70"
)

print(
    f"Final risk score     : "
    f"{highest_result['risk_score']:.2f}/100"
)

print(
    f"Risk level           : "
    f"{highest_result['risk_level']}"
)

print(
    f"Recommended action   : "
    f"{highest_result['recommended_action']}"
)


print("\nWhy was this transaction flagged?")

for number, reason in enumerate(
    highest_result["reasons"],
    start=1
):

    print(
        f"{number}. {reason}"
    )


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FRICTIONGUARD AI - RISK ENGINE COMPLETE")
print("=" * 75)

print(
    f"Transactions analyzed : "
    f"{len(sample_df):,}"
)

print(
    f"Risk results saved to  : "
    f"{OUTPUT_PATH}"
)

print("\nPipeline:")

print(
    "Transaction"
    " -> ML Prediction"
    " -> Behavioral Analysis"
    " -> Risk Score"
    " -> Explainable Decision"
)

print("\n" + "=" * 75)
print("READY FOR FRAUD DECISION SYSTEM")
print("=" * 75)