# 🛡️ FrictionGuard AI

### Explainable AI Fraud Detection & Risk Intelligence Platform

FrictionGuard AI is an AI-powered payment risk management system designed to detect suspicious financial transactions, explain the reasons behind risk decisions, and recommend defensive actions.

It combines **machine learning, behavioral analytics, transaction history, and rule-based risk signals** into a unified **0–100 risk score**.

> **Razorpay AI Buildathon 2026 — Track 2: AI Risk Manager**

---

## 🚀 Live Demo

### [🔗 Launch FrictionGuard AI](https://frictionguard-ai-kr8nbx9qsk8atiaxiqjbns.streamlit.app/)

Try the live:

* 🔍 Transaction Analysis
* 📊 Risk Intelligence
* 🚨 Fraud Alert Center
* 🧪 What-If Simulator
* ℹ️ System Information

---

## 🎯 Problem

Payment fraud detection is not only about predicting whether a transaction is fraudulent.

A practical risk system must also answer:

* **How risky is the transaction?**
* **Why was it flagged?**
* **How confident is the model?**
* **What action should be taken?**
* **How can false positives be minimized?**

FrictionGuard AI addresses these questions through an **explainable, behavioral, and defense-oriented risk decision pipeline**.

---

## 💡 Solution

FrictionGuard AI combines three key sources of evidence:

```text
┌──────────────────────────┐
│ Machine Learning         │
│ Fraud Probability        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Behavioral Risk Signals  │
│ + Transaction History    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Unified Risk Engine      │
│       0 – 100            │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Risk Classification      │
│ LOW / MEDIUM / HIGH /    │
│ CRITICAL                 │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Defensive Action         │
│ Allow / Verify / Review  │
│ / Block                  │
└──────────────────────────┘
```

The result is an **explainable risk decision rather than a simple black-box fraud label**.

---

# 🧠 Key Features

## 🔍 Live Transaction Analysis

Simulate a transaction and evaluate its fraud risk using:

* Transaction type
* Transaction amount
* Sender transaction history
* Receiver transaction history
* Historical average amounts
* Recent transaction activity
* Balance-related behavioral signals
* ML fraud probability

The system converts these signals into a unified risk assessment and recommended defensive action.

---

## 📊 Risk Intelligence

Provides an overview of detected transaction risk, including:

* Risk distribution
* Risk-level statistics
* High-risk transactions
* Transaction-level risk scores
* Fraud-risk patterns

This allows users to understand overall risk instead of examining transactions individually.

---

## 🚨 Fraud Alert Center

Provides a dedicated interface for investigating suspicious transactions.

Users can:

* Filter transactions by risk level
* Sort risk-scored transactions
* Identify potentially suspicious activity
* Prioritize transactions requiring investigation

---

## 🧪 What-If Simulator

The What-If Simulator allows users to modify transaction characteristics and observe how the final risk assessment changes.

```text
Change Transaction Characteristics
                ↓
        Behavioral Signals
                ↓
        ML Risk Assessment
                ↓
        Unified Risk Score
                ↓
        Risk Classification
```

This provides an interactive way to understand how different transaction and behavioral factors influence risk.

---

## 🛡️ Explainable Defensive Decisions

Each transaction receives:

* ML fraud probability
* Behavioral risk contribution
* Unified risk score
* Risk classification
* Recommended defensive response

### Risk Levels

| Risk Score | Level       | Defensive Action                |
| ---------: | ----------- | ------------------------------- |
|   **0–39** | 🟢 LOW      | ALLOW                           |
|  **40–64** | 🟡 MEDIUM   | ADDITIONAL VERIFICATION         |
|  **65–84** | 🟠 HIGH     | STEP-UP AUTHENTICATION / REVIEW |
| **85–100** | 🔴 CRITICAL | BLOCK / MANUAL INVESTIGATION    |

---

# 🤖 Machine Learning

The system uses a **Random Forest classifier** for fraud probability estimation.

The final decision combines the ML probability with behavioral risk signals.

```text
ML Fraud Probability
          +
Behavioral Risk Signals
          ↓
Unified Risk Score
          ↓
Risk Level
          ↓
Defensive Action
```

---

## 🔢 Engineered Features

The model uses transaction, historical, and behavioral information including:

* `step`
* `type`
* `amount`
* `sender_txn_count_before`
* `sender_avg_amount_before`
* `receiver_txn_count_before`
* `receiver_avg_amount_before`
* `sender_amount_vs_avg`
* `receiver_amount_vs_avg`
* `sender_txn_count_recent`
* `receiver_txn_count_recent`
* `sender_amount_recent`
* `receiver_amount_recent`
* `sender_old_balance`
* `receiver_old_balance`
* `sender_balance_to_amount`
* `receiver_balance_to_amount`
* `sender_zero_balance`
* `receiver_zero_balance`

These features capture transaction characteristics, historical behavior, recent activity, and balance-related signals.

---

# 📈 Final Model Performance

The final model was evaluated on an **untouched chronological test period**.

| Metric                  |      Result |
| ----------------------- | ----------: |
| **Precision**           |  **99.79%** |
| **Recall**              |  **85.61%** |
| **F1 Score**            |  **92.16%** |
| **PR-AUC**              |  **99.75%** |
| **ROC-AUC**             | **100.00%** |
| **False Positives**     |       **1** |
| **False-Positive Rate** | **0.0010%** |

### Test Confusion Matrix

|                   | Predicted Normal | Predicted Fraud |
| ----------------- | ---------------: | --------------: |
| **Actual Normal** |       **99,999** |           **1** |
| **Actual Fraud**  |           **80** |         **476** |

### Key Result

The final model achieved **99.79% precision** and **85.61% recall**, while producing only **1 false positive** in the evaluated test sample.

This demonstrates a strong focus on detecting fraudulent transactions while minimizing unnecessary friction for legitimate transactions.

---

# 🎯 False-Positive-Aware Threshold Selection

A key design decision in FrictionGuard AI is that the fraud classification threshold was **not simply fixed at 0.5**.

Instead, the threshold was selected using a validation period with an explicit focus on controlling false positives.

```text
Chronological Training Data
            ↓
       Fit Period
            ↓
   Validation Period
            ↓
Evaluate Candidate Thresholds
            ↓
Apply False-Positive Budget
            ↓
 Select Risk Threshold
            ↓
       Refit Model
            ↓
 Untouched Test Evaluation
```

The final Random Forest threshold was selected at:

### **0.88**

This reflects an important payment-risk principle:

> **Fraud detection must balance detection capability with customer friction.**

---

# ⏱️ Chronological Evaluation

To reduce the risk of temporal leakage, the dataset was divided chronologically rather than using a random train/test split.

```text
Earlier Transactions                         Later Transactions
──────────────────────────────────────────────────────────────►

      FIT              VALIDATION                 TEST
       64%                 16%                    20%
        │                   │                      │
        ▼                   ▼                      ▼
 Model Training     Threshold Selection     Final Evaluation
```

The final test period remained **untouched during model and threshold selection**.

For computational efficiency, the evaluation periods use a controlled number of normal transactions while retaining the fraud examples within each period.

---

# 🔐 Leakage-Aware Feature Engineering

The PaySim dataset contains balance-related fields that can potentially reveal information available only after a transaction.

FrictionGuard AI therefore avoids using **post-transaction balance fields as predictive inputs**.

Instead, the system uses historical and behavioral information such as:

* Previous transaction activity
* Historical average amounts
* Recent transaction behavior
* Historical balances
* Balance-to-amount relationships
* Zero-balance indicators

### Design Principle

**Risk decisions should be based on information available from transaction history and behavioral context rather than post-transaction outcomes.**

Recent transaction features represent previous transaction activity used for behavioral analysis.

---

# 📦 Dataset

FrictionGuard AI uses the **PaySim synthetic financial transaction dataset** for fraud-detection experimentation.

The dataset contains millions of synthetic financial transactions with a highly imbalanced fraud class.

Because the raw dataset is large, it is intentionally **not included in this GitHub repository**.

The repository contains the processed risk-scored demonstration data and trained model required by the deployed application.

---

# 🏗️ System Architecture

```text
                         ┌───────────────────┐
                         │ Transaction Input │
                         └─────────┬─────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │ Feature Engineering      │
                    │ Historical + Behavioral  │
                    │ Features                 │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
          ┌───────────────────┐     ┌───────────────────┐
          │ Random Forest     │     │ Behavioral Risk   │
          │ Fraud Model       │     │ Engine            │
          └─────────┬─────────┘     └─────────┬─────────┘
                    │                         │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Unified Risk Score       │
                    │        0 – 100           │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Risk Classification      │
                    │ LOW / MEDIUM / HIGH /    │
                    │ CRITICAL                 │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Defensive Response        │
                    │ Allow / Verify / Review  │
                    │ / Block                  │
                    └──────────────────────────┘
```

---

# 🗂️ Project Structure

```text
FrictionGuard-AI/
│
├── app/
│   └── dashboard.py
│
├── data/
│   └── processed/
│       └── risk_scored_transactions.csv
│
├── models/
│   ├── frictionguard_fraud_model.joblib
│   ├── fraud_threshold.txt
│   └── model_results.csv
│
├── src/
│   ├── data/
│   │   ├── full_fraud_profile.py
│   │   ├── inspect_dataset.py
│   │   └── leakage_analysis.py
│   │
│   ├── decision/
│   │   └── risk_engine.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   └── train_model.py
│   │
│   └── utils/
│
├── tests/
│
├── docs/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# ⚙️ Technology Stack

| Technology        | Purpose                      |
| ----------------- | ---------------------------- |
| **Python**        | Core development             |
| **Streamlit**     | Interactive risk dashboard   |
| **Pandas**        | Data processing              |
| **NumPy**         | Numerical computation        |
| **Scikit-learn**  | Machine learning             |
| **Random Forest** | Fraud probability estimation |
| **Joblib**        | Model persistence            |

---

# ▶️ Run Locally

## 1. Clone the repository

```bash
git clone https://github.com/kruthikasrinivasa/FrictionGuard-AI.git
cd FrictionGuard-AI
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

## 3. Activate the environment

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Launch the dashboard

```bash
streamlit run app/dashboard.py
```

The application will open in your browser.

---

# 🛡️ Defense-Only Design

FrictionGuard AI is strictly designed as a **defensive fraud-risk management system**.

The system focuses on:

* Fraud detection
* Risk assessment
* Explainability
* Additional verification
* Manual investigation
* Defensive blocking

It does **not** provide methods for committing fraud, bypassing payment security, or evading fraud-detection systems.

---

# 🏆 Razorpay AI Buildathon Alignment

## Track 2 — AI Risk Manager

FrictionGuard AI addresses the track through:

| Buildathon Requirement       | FrictionGuard AI                         |
| ---------------------------- | ---------------------------------------- |
| **Fraud / loss detection**   | Random Forest fraud detector             |
| **Risk assessment**          | Unified 0–100 risk score                 |
| **Explainability**           | ML probability + behavioral risk signals |
| **False-positive awareness** | Validation-based threshold selection     |
| **Defensive response**       | Allow / Verify / Review / Block          |
| **Quantitative evaluation**  | Precision, Recall, F1, PR-AUC, ROC-AUC   |
| **Held-out evaluation**      | Untouched chronological test period      |
| **Working demonstration**    | Interactive Streamlit dashboard          |
| **Defense-only design**      | Fraud detection and risk management      |

---

# 🔮 Future Improvements

Potential production extensions include:

* Real-time transaction streaming
* Online model monitoring
* Concept-drift detection
* Graph-based fraud detection
* Advanced anomaly detection
* Human-in-the-loop investigation workflows
* Cost-sensitive threshold optimization
* Advanced model explainability
* Production payment-gateway integration

---

# ⚠️ Limitations

FrictionGuard AI is currently a **research and demonstration system**.

Current limitations include:

* Evaluation uses the PaySim synthetic dataset.
* The deployed application is not connected to a live payment gateway.
* Performance on synthetic data does not guarantee equivalent performance on real-world payment data.
* Production deployment would require additional security, monitoring, latency, scalability, and compliance controls.

These limitations are stated explicitly to keep the project evaluation transparent.

---

# 📌 Project Status

| Component              | Status     |
| ---------------------- | ---------- |
| Feature Engineering    | ✅ Complete |
| Fraud Model            | ✅ Complete |
| Threshold Optimization | ✅ Complete |
| Risk Engine            | ✅ Complete |
| Streamlit Dashboard    | ✅ Complete |
| What-If Simulator      | ✅ Complete |
| Fraud Alert Center     | ✅ Complete |
| Risk Intelligence      | ✅ Complete |
| GitHub Repository      | ✅ Public   |
| Live Deployment        | ✅ Online   |

---

# 👩‍💻 FrictionGuard AI

### Explainable AI Risk Manager for Payment Fraud

Built for the **Razorpay AI Buildathon 2026 — Track 2: AI Risk Manager**.

### 🔗 Links

**🚀 Live Demo**

https://frictionguard-ai-kr8nbx9qsk8atiaxiqjbns.streamlit.app/

**💻 GitHub Repository**

https://github.com/kruthikasrinivasa/FrictionGuard-AI

---

<p align="center">

### 🛡️ FrictionGuard AI

**Detect suspicious activity. Explain the risk. Reduce unnecessary friction.**

</p>
