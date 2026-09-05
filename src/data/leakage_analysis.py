from pathlib import Path
import pandas as pd


# ============================================================
# FrictionGuard AI
# Pre-Transaction Data & Leakage Analysis
# ============================================================

DATA_PATH = Path(
    "data/raw/PS_20174392719_1491204439457_log.csv"
)


def main():

    print("=" * 70)
    print("FRICTIONGUARD AI — PRE-TRANSACTION DATA ANALYSIS")
    print("=" * 70)

    columns = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    ]

    df = pd.read_csv(
        DATA_PATH,
        usecols=columns
    )

    print(f"\nLoaded {len(df):,} transactions.")

    # --------------------------------------------------------
    # Column classification
    # --------------------------------------------------------

    pre_transaction = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "oldbalanceOrg",
        "oldbalanceDest",
    ]

    post_transaction_or_risk_flags = [
        "newbalanceOrig",
        "newbalanceDest",
        "isFlaggedFraud",
    ]

    target = [
        "isFraud"
    ]

    print("\n" + "-" * 70)
    print("PRE-TRANSACTION CANDIDATES")
    print("-" * 70)

    for column in pre_transaction:
        print(f"✓ {column}")

    print("\n" + "-" * 70)
    print("POTENTIAL LEAKAGE / POST-TRANSACTION FEATURES")
    print("-" * 70)

    for column in post_transaction_or_risk_flags:
        print(f"⚠ {column}")

    print("\n" + "-" * 70)
    print("TARGET")
    print("-" * 70)

    for column in target:
        print(f"🎯 {column}")

    # --------------------------------------------------------
    # Balance consistency analysis
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("BALANCE CONSISTENCY")
    print("-" * 70)

    df["origin_balance_change"] = (
        df["oldbalanceOrg"] - df["amount"]
    )

    df["destination_balance_change"] = (
        df["oldbalanceDest"] + df["amount"]
    )

    origin_difference = (
        df["origin_balance_change"]
        - df["newbalanceOrig"]
    ).abs()

    destination_difference = (
        df["destination_balance_change"]
        - df["newbalanceDest"]
    ).abs()

    print(
        "Origin balance mismatches:",
        (origin_difference > 0.01).sum()
    )

    print(
        "Destination balance mismatches:",
        (destination_difference > 0.01).sum()
    )

    # --------------------------------------------------------
    # Fraud distribution
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("FRAUD BY TRANSACTION TYPE")
    print("-" * 70)

    fraud_summary = (
        df.groupby("type")["isFraud"]
        .agg(["count", "sum", "mean"])
        .sort_values("sum", ascending=False)
    )

    fraud_summary["fraud_percentage"] = (
        fraud_summary["mean"] * 100
    )

    print(fraud_summary)

    # --------------------------------------------------------
    # Flagged fraud analysis
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("isFlaggedFraud ANALYSIS")
    print("-" * 70)

    print(
        pd.crosstab(
            df["isFlaggedFraud"],
            df["isFraud"],
            margins=True
        )
    )

    print("\n" + "=" * 70)
    print("LEAKAGE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()