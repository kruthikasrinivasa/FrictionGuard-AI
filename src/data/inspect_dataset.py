from pathlib import Path
import pandas as pd


# ============================================================
# FrictionGuard AI
# Dataset Inspection — PaySim
# ============================================================

DATA_PATH = Path(
    "data/raw/PS_20174392719_1491204439457_log.csv"
)


def main():

    print("=" * 70)
    print("FRICTIONGUARD AI — DATASET INSPECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. File information
    # --------------------------------------------------------

    file_size_mb = DATA_PATH.stat().st_size / (1024 ** 2)

    print(f"\nFile: {DATA_PATH.name}")
    print(f"File size: {file_size_mb:.2f} MB")

    # --------------------------------------------------------
    # 2. Read only the header first
    # --------------------------------------------------------

    df_header = pd.read_csv(DATA_PATH, nrows=5)

    print("\n" + "-" * 70)
    print("COLUMNS")
    print("-" * 70)

    for i, column in enumerate(df_header.columns, start=1):
        print(f"{i:2}. {column}")

    # --------------------------------------------------------
    # 3. Sample rows
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("SAMPLE DATA")
    print("-" * 70)

    print(df_header.to_string(index=False))

    # --------------------------------------------------------
    # 4. Dataset shape
    # --------------------------------------------------------
    # We count rows without loading the entire dataset.

    print("\n" + "-" * 70)
    print("ROW COUNT")
    print("-" * 70)

    with open(DATA_PATH, "rb") as f:
        row_count = sum(1 for _ in f) - 1

    print(f"Rows: {row_count:,}")
    print(f"Columns: {len(df_header.columns)}")

    # --------------------------------------------------------
    # 5. Data types
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("DATA TYPES")
    print("-" * 70)

    print(df_header.dtypes)

    # --------------------------------------------------------
    # 6. Fraud distribution
    # --------------------------------------------------------
    # Read only the target column.

    fraud = pd.read_csv(
        DATA_PATH,
        usecols=["isFraud"]
    )

    print("\n" + "-" * 70)
    print("FRAUD DISTRIBUTION")
    print("-" * 70)

    counts = fraud["isFraud"].value_counts()
    percentages = fraud["isFraud"].value_counts(normalize=True) * 100

    print(f"Legitimate (0): {counts.get(0, 0):,}")
    print(f"Fraudulent  (1): {counts.get(1, 0):,}")

    print(
        f"\nFraud percentage: "
        f"{percentages.get(1, 0):.4f}%"
    )

    # --------------------------------------------------------
    # 7. Missing values
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("MISSING VALUES")
    print("-" * 70)

    sample = pd.read_csv(
        DATA_PATH,
        nrows=100_000
    )

    missing = sample.isnull().sum()

    print(
        missing[missing > 0]
        if missing.sum() > 0
        else "No missing values found in first 100,000 rows."
    )

    # --------------------------------------------------------
    # 8. Duplicate rows in sample
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("DUPLICATES")
    print("-" * 70)

    duplicate_count = sample.duplicated().sum()

    print(
        f"Duplicate rows in first 100,000 rows: "
        f"{duplicate_count:,}"
    )

    # --------------------------------------------------------
    # 9. Transaction types
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("TRANSACTION TYPES")
    print("-" * 70)

    print(sample["type"].value_counts())

    # --------------------------------------------------------
    # 10. Fraud by transaction type
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("FRAUD BY TRANSACTION TYPE")
    print("-" * 70)

    fraud_by_type = (
        sample
        .groupby("type")["isFraud"]
        .agg(["count", "sum", "mean"])
        .sort_values("sum", ascending=False)
    )

    fraud_by_type["fraud_percentage"] = (
        fraud_by_type["mean"] * 100
    )

    print(fraud_by_type)

    print("\n" + "=" * 70)
    print("DATASET INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()