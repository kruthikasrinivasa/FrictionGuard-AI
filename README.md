# 🛡️ FrictionGuard AI

### Explainable AI Fraud Detection & Risk Intelligence Platform

FrictionGuard AI is an AI-powered payment risk management system designed to detect suspicious financial transactions, explain the reasons behind risk decisions, and recommend defensive actions.

It combines machine learning, behavioral analytics, transaction history, and rule-based risk signals into a unified **0–100 risk score**.

> **Razorpay AI Buildathon 2026 — Track 2: AI Risk Manager**

---

## 🚀 Live Demo

🔗 **[Launch FrictionGuard AI](https://frictionguard-ai-kr8nbx9qsk8atiaxiqjbns.streamlit.app/)**

Try the live transaction analysis, risk intelligence dashboard, fraud alert center, and What-If simulator.

---

## 🎯 Problem

Payment fraud detection is not only about predicting whether a transaction is fraudulent.

A practical risk system must also answer:

- How risky is the transaction?
- Why was it flagged?
- How confident is the model?
- What action should be taken?
- How can false positives be minimized?

FrictionGuard AI addresses these questions through an explainable and defense-oriented risk decision pipeline.

---

## 💡 Solution

FrictionGuard AI combines:

**Machine Learning Probability**

+

**Behavioral Risk Signals**

+

**Transaction History**

↓

**Unified Risk Score (0–100)**

↓

**Risk Level**

↓

**Defensive Action**

---

## 🧠 Key Features

### 🔍 Live Transaction Analysis

Simulate a transaction and evaluate its fraud risk using:

- Transaction type
- Transaction amount
- Sender transaction history
- Receiver transaction history
- Historical average amounts
- Recent transaction activity
- Balance-related behavioral signals
- ML fraud probability

---

### 📊 Risk Intelligence

Provides an overview of detected transaction risk, including:

- Risk distribution
- Risk-level statistics
- High-risk transactions
- Transaction-level risk scores

---

### 🚨 Fraud Alert Center

Provides a dedicated view for investigating suspicious transactions.

Users can filter and sort risk-scored transactions to identify potentially fraudulent activity.

---

### 🧪 What-If Simulator

Test how changing transaction characteristics affects the final risk score.

This helps demonstrate how different behavioral signals influence risk decisions.

---

### 🛡️ Explainable Defensive Decisions

Each transaction receives:

- ML fraud probability
- Behavioral risk contribution
- Unified risk score
- Risk classification
- Recommended defensive response

Risk levels:

| Risk Score | Level | Defensive Action |
|---:|---|---|
| 0–39 | 🟢 LOW | ALLOW |
| 40–64 | 🟡 MEDIUM | ADDITIONAL VERIFICATION |
| 65–84 | 🟠 HIGH | STEP-UP AUTHENTICATION / REVIEW |
| 85–100 | 🔴 CRITICAL | BLOCK / MANUAL INVESTIGATION |

---

## 🤖 Machine Learning

The system uses a **Random Forest classifier** for fraud probability estimation.

The final decision combines the ML probability with behavioral risk signals.

### Final model evaluation

The model was evaluated on an **untouched chronological test period**.

| Metric | Result |
|---|---:|
| Precision | **99.79%** |
| Recall | **85.61%** |
| F1 Score | **92.16%** |
| PR-AUC | **99.75%** |
| ROC-AUC | **100.00%** |
| False Positives | **1** |
| False-Positive Rate | **0.0010%** |

### Test confusion matrix

```text
                    Predicted
                  Normal  Fraud

Actual Normal      99,999      1
Actual Fraud           80    476