# 🛡️ FrictionGuard AI

### Explainable AI Risk Manager for Payment Fraud

**FrictionGuard AI** is an AI-powered payment risk management platform that detects suspicious financial transactions, combines machine-learning predictions with behavioral signals, explains why a transaction is risky, and recommends a defensive action.

> **Razorpay AI Buildathon 2026 — Track 2: AI Risk Manager**

<p align="center">

**🔍 Detect** → **🧠 Explain** → **📊 Score** → **🛡️ Respond**

</p>

---

## 🚀 Live Demo

### 👉 [Launch FrictionGuard AI](https://frictionguard-ai-kr8nbx9qsk8atiaxiqjbns.streamlit.app/)

Try the complete interactive system:

- 🔍 Live Transaction Analysis
- 📊 Risk Intelligence
- 🚨 Fraud Alert Center
- 🧪 What-If Risk Simulator
- ℹ️ System Information

---

## 🎯 Problem

Payment fraud detection is not simply a binary classification problem.

A practical risk-management system must answer:

- **How risky is this transaction?**
- **Why is it risky?**
- **How confident is the ML model?**
- **What defensive action should be taken?**
- **Can legitimate customers be protected from unnecessary friction?**

Traditional threshold-based classification can focus heavily on fraud recall while overlooking the operational cost of false positives.

**FrictionGuard AI is designed around the complete risk-decision process rather than prediction alone.**

---

# 💡 Our Approach

FrictionGuard combines three sources of evidence:

```text
                 ┌──────────────────────┐
                 │ Machine Learning      │
                 │ Fraud Probability    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Behavioral Signals   │
                 │ Transaction History  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Unified Risk Engine  │
                 │      0 – 100         │
                 └──────────┬───────────┘
                            │
                            ▼
             ┌──────────────────────────────┐
             │ Risk Classification          │
             │ LOW / MEDIUM / HIGH /        │
             │ CRITICAL                     │
             └──────────────┬───────────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Defensive Response  │
                 │ Allow / Verify /     │
                 │ Review / Block       │
                 └──────────────────────┘