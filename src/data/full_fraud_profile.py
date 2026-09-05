from pathlib import Path
import pandas as pd


DATA_PATH = Path(
    "data/raw/PS_20174392719_1491204439457_log.csv"
)


def main():

    print("=" * 70)
    print("FRICTIONGUARD AI — FULL FRAUD PROFILE")
    print("=" * 70)

    df = pd.read_csv(
        DATA_PATH,
        usecols=["step", "type", "amount", "isFraud"]
    )

    print(f"\nTransactions loaded: {len(df):,}")

    # --------------------------------------------------------
    # Fraud by transaction type
    # --------------------------------------------------------

    profile = (
        df.groupby("type")["isFraud"]
        .agg(
            transactions="count",
            fraud_cases="sum"
        )
    )

    profile["fraud_rate_percent"] = (
        profile["fraud_cases"]
        / profile["transactions"]
        * 100
    )

    profile["fraud_share_percent"] = (
        profile["fraud_cases"]
        / df["isFraud"].sum()
        * 100
    )

    profile = profile.sort_values(
        "fraud_cases",
        ascending=False
    )

    print("\n" + "-" * 70)
    print("FRAUD PROFILE BY TRANSACTION TYPE")
    print("-" * 70)

    print(profile.to_string())

    # --------------------------------------------------------
    # Fraud by time step
    # --------------------------------------------------------

    time_profile = (
        df.groupby("step")["isFraud"]
        .agg(
            transactions="count",
            fraud_cases="sum"
        )
    )

    time_profile["fraud_rate_percent"] = (
        time_profile["fraud_cases"]
        / time_profile["transactions"]
        * 100
    )

    highest_risk_steps = (
        time_profile
        .sort_values(
            "fraud_rate_percent",
            ascending=False
        )
        .head(10)
    )

    print("\n" + "-" * 70)
    print("TOP 10 HIGHEST-FRAUD-RATE TIME STEPS")
    print("-" * 70)

    print(highest_risk_steps.to_string())

    print("\n" + "=" * 70)
    print("FULL FRAUD PROFILE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()