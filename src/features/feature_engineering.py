import pandas as pd
from pathlib import Path
from collections import defaultdict, deque
import gc


# ============================================================
# FRICTIONGUARD AI - FULL FEATURE ENGINEERING
# MEMORY-EFFICIENT VERSION
# ============================================================

print("=" * 70)
print("FRICTIONGUARD AI - FULL FEATURE ENGINEERING")
print("=" * 70)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_PATH = (
    PROCESSED_DIR / "engineered_features.csv"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print(f"Dataset: {DATA_PATH}")
print(f"Dataset exists: {DATA_PATH.exists()}")
print(f"Output: {OUTPUT_PATH}")


if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Smaller chunks = lower RAM usage
CHUNK_SIZE = 25_000

# Number of recent transactions to remember
RECENT_WINDOW = 10


# ============================================================
# 3. HISTORY STORAGE
# ============================================================

# Sender historical information
sender_counts = defaultdict(int)
sender_amount_sum = defaultdict(float)

# Receiver historical information
receiver_counts = defaultdict(int)
receiver_amount_sum = defaultdict(float)


# Recent sender transactions
sender_recent_history = defaultdict(deque)

# Recent receiver transactions
receiver_recent_history = defaultdict(deque)


# ============================================================
# 4. FEATURES
# ============================================================

FEATURE_COLUMNS = [
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
# 5. OUTPUT COLUMNS
# ============================================================

OUTPUT_COLUMNS = [
    "step",
    "type",
    "amount",
    "isFraud"
] + FEATURE_COLUMNS


# ============================================================
# 6. COUNTERS
# ============================================================

total_rows = 0
fraud_rows = 0
chunk_number = 0


# ============================================================
# 7. REMOVE OLD OUTPUT
# ============================================================

if OUTPUT_PATH.exists():

    print()
    print("Removing old incomplete processed dataset...")

    OUTPUT_PATH.unlink()

    print("Old file removed.")


# ============================================================
# 8. START
# ============================================================

print()
print("=" * 70)
print("STARTING FULL DATASET PROCESSING")
print("=" * 70)

print(
    f"Chunk size: {CHUNK_SIZE:,} rows"
)

print(
    f"Recent transaction window: {RECENT_WINDOW}"
)

print()


# ============================================================
# 9. READ DATA IN CHUNKS
# ============================================================

for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=CHUNK_SIZE
):

    chunk_number += 1

    print("-" * 70)
    print(
        f"Processing chunk {chunk_number}..."
    )


    # ========================================================
    # 10. CONVERT NUMERIC COLUMNS
    # ========================================================

    chunk["amount"] = pd.to_numeric(
        chunk["amount"],
        errors="coerce"
    ).fillna(0).astype("float32")


    chunk["oldbalanceOrg"] = pd.to_numeric(
        chunk["oldbalanceOrg"],
        errors="coerce"
    ).fillna(0).astype("float32")


    chunk["oldbalanceDest"] = pd.to_numeric(
        chunk["oldbalanceDest"],
        errors="coerce"
    ).fillna(0).astype("float32")


    # ========================================================
    # 11. FEATURE ARRAYS
    # ========================================================

    sender_txn_count_before = []
    sender_avg_amount_before = []

    receiver_txn_count_before = []
    receiver_avg_amount_before = []

    sender_amount_vs_avg = []
    receiver_amount_vs_avg = []

    sender_txn_count_recent = []
    receiver_txn_count_recent = []

    sender_amount_recent = []
    receiver_amount_recent = []

    sender_old_balance = []
    receiver_old_balance = []

    sender_balance_to_amount = []
    receiver_balance_to_amount = []

    sender_zero_balance = []
    receiver_zero_balance = []


    # ========================================================
    # 12. PROCESS EACH TRANSACTION
    # ========================================================

    for sender, receiver, amount, old_sender_balance, old_receiver_balance, fraud in zip(
        chunk["nameOrig"],
        chunk["nameDest"],
        chunk["amount"],
        chunk["oldbalanceOrg"],
        chunk["oldbalanceDest"],
        chunk["isFraud"]
    ):

        amount = float(amount)
        old_sender_balance = float(old_sender_balance)
        old_receiver_balance = float(old_receiver_balance)


        # ====================================================
        # SENDER HISTORY
        # ====================================================

        sender_count = sender_counts[sender]

        if sender_count > 0:

            sender_avg = (
                sender_amount_sum[sender]
                / sender_count
            )

        else:

            sender_avg = 0.0


        sender_txn_count_before.append(
            sender_count
        )

        sender_avg_amount_before.append(
            sender_avg
        )


        # ====================================================
        # RECEIVER HISTORY
        # ====================================================

        receiver_count = receiver_counts[receiver]

        if receiver_count > 0:

            receiver_avg = (
                receiver_amount_sum[receiver]
                / receiver_count
            )

        else:

            receiver_avg = 0.0


        receiver_txn_count_before.append(
            receiver_count
        )

        receiver_avg_amount_before.append(
            receiver_avg
        )


        # ====================================================
        # AMOUNT VS HISTORICAL AVERAGE
        # ====================================================

        if sender_avg > 0:

            sender_deviation = (
                amount / sender_avg
            )

        else:

            sender_deviation = 0.0


        if receiver_avg > 0:

            receiver_deviation = (
                amount / receiver_avg
            )

        else:

            receiver_deviation = 0.0


        sender_amount_vs_avg.append(
            sender_deviation
        )

        receiver_amount_vs_avg.append(
            receiver_deviation
        )


        # ====================================================
        # RECENT SENDER HISTORY
        # ====================================================

        sender_history = (
            sender_recent_history[sender]
        )

        sender_recent_count = len(
            sender_history
        )

        sender_recent_amount = sum(
            value
            for value in sender_history
        )


        sender_txn_count_recent.append(
            sender_recent_count
        )

        sender_amount_recent.append(
            sender_recent_amount
        )


        # ====================================================
        # RECENT RECEIVER HISTORY
        # ====================================================

        receiver_history = (
            receiver_recent_history[receiver]
        )

        receiver_recent_count = len(
            receiver_history
        )

        receiver_recent_amount = sum(
            value
            for value in receiver_history
        )


        receiver_txn_count_recent.append(
            receiver_recent_count
        )

        receiver_amount_recent.append(
            receiver_recent_amount
        )


        # ====================================================
        # PRE-TRANSACTION BALANCES
        # ====================================================

        sender_old_balance.append(
            old_sender_balance
        )

        receiver_old_balance.append(
            old_receiver_balance
        )


        # ====================================================
        # BALANCE / AMOUNT RATIO
        # ====================================================

        if amount > 0:

            sender_ratio = (
                old_sender_balance / amount
            )

            receiver_ratio = (
                old_receiver_balance / amount
            )

        else:

            sender_ratio = 0.0
            receiver_ratio = 0.0


        sender_balance_to_amount.append(
            sender_ratio
        )

        receiver_balance_to_amount.append(
            receiver_ratio
        )


        # ====================================================
        # ZERO BALANCE FLAGS
        # ====================================================

        sender_zero_balance.append(
            int(old_sender_balance == 0)
        )

        receiver_zero_balance.append(
            int(old_receiver_balance == 0)
        )


        # ====================================================
        # UPDATE HISTORY
        #
        # IMPORTANT:
        # Current transaction is added AFTER
        # calculating its features.
        #
        # This prevents target/data leakage.
        # ====================================================

        sender_counts[sender] += 1

        sender_amount_sum[sender] += amount


        receiver_counts[receiver] += 1

        receiver_amount_sum[receiver] += amount


        # ====================================================
        # UPDATE RECENT HISTORY
        # ====================================================

        sender_history.append(
            amount
        )

        receiver_history.append(
            amount
        )


        while len(sender_history) > RECENT_WINDOW:

            sender_history.popleft()


        while len(receiver_history) > RECENT_WINDOW:

            receiver_history.popleft()


        # ====================================================
        # GLOBAL COUNTERS
        # ====================================================

        total_rows += 1

        if int(fraud) == 1:

            fraud_rows += 1


    # ========================================================
    # 13. ADD FEATURES TO CHUNK
    # ========================================================

    chunk["sender_txn_count_before"] = (
        sender_txn_count_before
    )

    chunk["sender_avg_amount_before"] = (
        sender_avg_amount_before
    )

    chunk["receiver_txn_count_before"] = (
        receiver_txn_count_before
    )

    chunk["receiver_avg_amount_before"] = (
        receiver_avg_amount_before
    )

    chunk["sender_amount_vs_avg"] = (
        sender_amount_vs_avg
    )

    chunk["receiver_amount_vs_avg"] = (
        receiver_amount_vs_avg
    )

    chunk["sender_txn_count_recent"] = (
        sender_txn_count_recent
    )

    chunk["receiver_txn_count_recent"] = (
        receiver_txn_count_recent
    )

    chunk["sender_amount_recent"] = (
        sender_amount_recent
    )

    chunk["receiver_amount_recent"] = (
        receiver_amount_recent
    )

    chunk["sender_old_balance"] = (
        sender_old_balance
    )

    chunk["receiver_old_balance"] = (
        receiver_old_balance
    )

    chunk["sender_balance_to_amount"] = (
        sender_balance_to_amount
    )

    chunk["receiver_balance_to_amount"] = (
        receiver_balance_to_amount
    )

    chunk["sender_zero_balance"] = (
        sender_zero_balance
    )

    chunk["receiver_zero_balance"] = (
        receiver_zero_balance
    )


    # ========================================================
    # 14. SELECT ONLY REQUIRED COLUMNS
    # ========================================================

    output_chunk = chunk.loc[
        :,
        OUTPUT_COLUMNS
    ]


    # ========================================================
    # 15. SAVE
    # ========================================================

    output_chunk.to_csv(
        OUTPUT_PATH,
        mode="a",
        header=(chunk_number == 1),
        index=False
    )


    # ========================================================
    # 16. PROGRESS
    # ========================================================

    file_size_mb = (
        OUTPUT_PATH.stat().st_size
        / (1024 * 1024)
    )


    print(
        f"Rows processed: {total_rows:,}"
    )

    print(
        f"Fraud rows seen: {fraud_rows:,}"
    )

    print(
        f"Processed file size: {file_size_mb:.2f} MB"
    )


    # ========================================================
    # 17. RELEASE MEMORY
    # ========================================================

    del chunk
    del output_chunk

    del sender_txn_count_before
    del sender_avg_amount_before

    del receiver_txn_count_before
    del receiver_avg_amount_before

    del sender_amount_vs_avg
    del receiver_amount_vs_avg

    del sender_txn_count_recent
    del receiver_txn_count_recent

    del sender_amount_recent
    del receiver_amount_recent

    del sender_old_balance
    del receiver_old_balance

    del sender_balance_to_amount
    del receiver_balance_to_amount

    del sender_zero_balance
    del receiver_zero_balance

    gc.collect()


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"Total rows processed: {total_rows:,}"
)

print(
    f"Total fraud rows: {fraud_rows:,}"
)


if total_rows > 0:

    fraud_rate = (
        fraud_rows / total_rows
    ) * 100

    print(
        f"Overall fraud rate: {fraud_rate:.6f}%"
    )


print()
print("=" * 70)
print("FEATURES CREATED")
print("=" * 70)


for number, feature in enumerate(
    FEATURE_COLUMNS,
    start=1
):

    print(
        f"{number:2d}. {feature}"
    )


print()
print("=" * 70)
print("DATA LEAKAGE PROTECTION: ENABLED")
print("=" * 70)

print()
print("Processed dataset:")
print(OUTPUT_PATH)

print()

print(
    "The engineered dataset is ready for ML preprocessing."
)

print("=" * 70)