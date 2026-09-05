import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "frictionguard_fraud_model.joblib"
)

RISK_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "risk_scored_transactions.csv"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FrictionGuard AI",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 30px;
    }

    .risk-box {
        padding: 22px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.3);
        margin-top: 10px;
    }

    .reason {
        padding: 10px;
        margin: 6px 0;
        border-radius: 8px;
        background: rgba(128,128,128,0.12);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ FrictionGuard AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Explainable AI Fraud Detection & Risk Intelligence Platform'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_risk_data():

    if not RISK_DATA_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(RISK_DATA_PATH)


model = load_model()
risk_data = load_risk_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛡️ FrictionGuard")

st.sidebar.markdown(
    """
    **AI Fraud Intelligence**

    Detects suspicious financial transactions using:

    • Machine Learning  
    • Behavioral Analytics  
    • Transaction History  
    • Risk Scoring  
    • Explainable Decisions  
    • Defense-Only Response
    """
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🔍 Live Transaction Analysis",
        "📊 Risk Intelligence",
        "🚨 Fraud Alert Center",
        "🧪 What-If Simulator",
        "ℹ️ System Information",
    ],
)


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURES = [
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
# RISK ENGINE
# ============================================================

def analyze_transaction(
    step,
    transaction_type,
    amount,
    sender_txn_before,
    sender_avg,
    receiver_txn_before,
    receiver_avg,
    sender_recent,
    receiver_recent,
    sender_recent_amount,
    receiver_recent_amount,
    sender_old_balance,
    receiver_old_balance,
):

    # Match feature_engineering.py exactly:
    # when there is no prior history, the historical average is 0
    # and the amount-vs-average feature is also 0.
    if sender_txn_before > 0 and sender_avg > 0:
        sender_amount_vs_avg = amount / sender_avg
    else:
        sender_amount_vs_avg = 0.0

    if receiver_txn_before > 0 and receiver_avg > 0:
        receiver_amount_vs_avg = amount / receiver_avg
    else:
        receiver_amount_vs_avg = 0.0

    # These are the SUM of the previous recent transactions,
    # matching feature_engineering.py.
    sender_amount_recent = sender_recent_amount
    receiver_amount_recent = receiver_recent_amount

    sender_balance_to_amount = (
        sender_old_balance / max(amount, 0.01)
    )

    receiver_balance_to_amount = (
        receiver_old_balance / max(amount, 0.01)
    )

    sender_zero_balance = int(
        sender_old_balance == 0
    )

    receiver_zero_balance = int(
        receiver_old_balance == 0
    )


    transaction = pd.DataFrame(
        [{
            "step": step,
            "type": transaction_type,
            "amount": amount,

            "sender_txn_count_before":
                sender_txn_before,

            "sender_avg_amount_before":
                sender_avg,

            "receiver_txn_count_before":
                receiver_txn_before,

            "receiver_avg_amount_before":
                receiver_avg,

            "sender_amount_vs_avg":
                sender_amount_vs_avg,

            "receiver_amount_vs_avg":
                receiver_amount_vs_avg,

            "sender_txn_count_recent":
                sender_recent,

            "receiver_txn_count_recent":
                receiver_recent,

            "sender_amount_recent":
                sender_amount_recent,

            "receiver_amount_recent":
                receiver_amount_recent,

            "sender_old_balance":
                sender_old_balance,

            "receiver_old_balance":
                receiver_old_balance,

            "sender_balance_to_amount":
                sender_balance_to_amount,

            "receiver_balance_to_amount":
                receiver_balance_to_amount,

            "sender_zero_balance":
                sender_zero_balance,

            "receiver_zero_balance":
                receiver_zero_balance,
        }]
    )


    probability = float(
        model.predict_proba(
            transaction[FEATURES]
        )[0][1]
    )


    # --------------------------------------------------------
    # Behavioral score
    # --------------------------------------------------------

    behavioral_score = 0
    reasons = []


    if sender_amount_vs_avg >= 10:

        behavioral_score += 18

        reasons.append(
            "Transaction amount is extremely high "
            "compared with sender history."
        )

    elif sender_amount_vs_avg >= 5:

        behavioral_score += 12

        reasons.append(
            "Transaction amount is significantly "
            "higher than sender's normal pattern."
        )

    elif sender_amount_vs_avg >= 3:

        behavioral_score += 7

        reasons.append(
            "Transaction amount is above sender's "
            "historical pattern."
        )


    if receiver_amount_vs_avg >= 10:

        behavioral_score += 15

        reasons.append(
            "Amount is highly unusual for the receiver."
        )

    elif receiver_amount_vs_avg >= 5:

        behavioral_score += 10

        reasons.append(
            "Receiver is handling an unusually large amount."
        )


    if sender_recent >= 10:

        behavioral_score += 12

        reasons.append(
            "High recent sender transaction activity."
        )

    elif sender_recent >= 5:

        behavioral_score += 7

        reasons.append(
            "Elevated recent sender activity."
        )


    if receiver_recent >= 15:

        behavioral_score += 10

        reasons.append(
            "Receiver shows unusually high recent activity."
        )

    elif receiver_recent >= 8:

        behavioral_score += 5

        reasons.append(
            "Receiver has elevated recent activity."
        )


    if (
        sender_old_balance > 0
        and amount > sender_old_balance
    ):

        behavioral_score += 15

        reasons.append(
            "Transaction amount exceeds sender's "
            "available balance."
        )


    if sender_zero_balance:

        behavioral_score += 8

        reasons.append(
            "Sender account has zero balance."
        )


    if receiver_zero_balance:

        behavioral_score += 5

        reasons.append(
            "Receiver account has zero balance."
        )


    # --------------------------------------------------------
    # Final risk
    # --------------------------------------------------------

    ml_score = probability * 60

    risk_score = min(
        100,
        ml_score + behavioral_score
    )


    if risk_score >= 85:

        level = "CRITICAL"
        action = "BLOCK / MANUAL INVESTIGATION"

    elif risk_score >= 65:

        level = "HIGH"
        action = "STEP-UP AUTHENTICATION / REVIEW"

    elif risk_score >= 40:

        level = "MEDIUM"
        action = "ADDITIONAL VERIFICATION"

    else:

        level = "LOW"
        action = "ALLOW"


    if probability >= 0.90:

        reasons.insert(
            0,
            "ML model assigns very high fraud probability."
        )

    elif probability >= 0.70:

        reasons.insert(
            0,
            "ML model detects elevated fraud probability."
        )

    if not reasons:

        reasons.append(
            "No significant anomalous behavior detected."
        )


    return {
        "probability": probability,
        "behavioral_score": behavioral_score,
        "risk_score": risk_score,
        "level": level,
        "action": action,
        "reasons": reasons[:5],
    }


# ============================================================
# PAGE 1 — LIVE ANALYSIS
# ============================================================

if page == "🔍 Live Transaction Analysis":

    st.header("🔍 Live Transaction Analysis")

    st.write(
        "Simulate a transaction and let FrictionGuard "
        "evaluate its fraud risk."
    )
    st.caption(
        "Recent amount totals represent the previous transactions only, "
        "matching the training feature definition."
    )

    st.divider()


    col1, col2, col3 = st.columns(3)


    with col1:

        transaction_type = st.selectbox(
            "Transaction Type",
            [
                "PAYMENT",
                "TRANSFER",
                "CASH_OUT",
                "CASH_IN",
                "DEBIT",
            ],
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.01,
            value=10000.0,
            step=1000.0,
        )

        step = st.number_input(
            "Transaction Step",
            min_value=1,
            value=1,
        )


    with col2:

        sender_txn_before = st.number_input(
            "Sender Previous Transactions",
            min_value=0,
            value=5,
        )

        sender_avg = st.number_input(
            "Sender Historical Avg Amount",
            min_value=0.01,
            value=10000.0,
        )

        sender_recent = st.number_input(
            "Sender Recent Transactions",
            min_value=0,
            value=2,
        )

        sender_recent_amount = st.number_input(
            "Sender Recent Amount Total",
            min_value=0.0,
            value=20000.0,
            step=1000.0,
            help="Sum of the sender's previous recent transaction amounts.",
        )


    with col3:

        receiver_txn_before = st.number_input(
            "Receiver Previous Transactions",
            min_value=0,
            value=5,
        )

        receiver_avg = st.number_input(
            "Receiver Historical Avg Amount",
            min_value=0.01,
            value=10000.0,
        )

        receiver_recent = st.number_input(
            "Receiver Recent Transactions",
            min_value=0,
            value=2,
        )

        receiver_recent_amount = st.number_input(
            "Receiver Recent Amount Total",
            min_value=0.0,
            value=20000.0,
            step=1000.0,
            help="Sum of the receiver's previous recent transaction amounts.",
        )


    col4, col5 = st.columns(2)


    with col4:

        sender_balance = st.number_input(
            "Sender Balance",
            min_value=0.0,
            value=25000.0,
            step=1000.0,
        )


    with col5:

        receiver_balance = st.number_input(
            "Receiver Balance",
            min_value=0.0,
            value=10000.0,
            step=1000.0,
        )


    st.divider()


    analyze = st.button(
        "🚨 ANALYZE TRANSACTION",
        use_container_width=True,
    )


    if analyze:

        if model is None:

            st.error(
                "Fraud detection model not found."
            )

        else:

            result = analyze_transaction(
                step,
                transaction_type,
                amount,
                sender_txn_before,
                sender_avg,
                receiver_txn_before,
                receiver_avg,
                sender_recent,
                receiver_recent,
                sender_recent_amount,
                receiver_recent_amount,
                sender_balance,
                receiver_balance,
            )


            st.divider()

            st.subheader(
                "🛡️ FrictionGuard Decision"
            )


            m1, m2, m3, m4 = st.columns(4)


            m1.metric(
                "ML Fraud Probability",
                f"{result['probability'] * 100:.2f}%",
            )

            m2.metric(
                "Behavior Score",
                f"{result['behavioral_score']}/83",
            )

            m3.metric(
                "Risk Score",
                f"{result['risk_score']:.1f}/100",
            )

            m4.metric(
                "Risk Level",
                result["level"],
            )


            st.progress(
                min(
                    result["risk_score"] / 100,
                    1.0
                )
            )


            st.markdown(
                f"""
                <div class="risk-box">

                ### 🎯 Recommended Action

                **{result['action']}**

                </div>
                """,
                unsafe_allow_html=True,
            )


            st.subheader(
                "🔎 Explainable Decision"
            )


            for reason in result["reasons"]:

                st.markdown(
                    f"""
                    <div class="reason">
                    ⚠️ {reason}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ============================================================
# PAGE 2 — RISK INTELLIGENCE
# ============================================================

elif page == "📊 Risk Intelligence":

    st.header("📊 Risk Intelligence Dashboard")

    if risk_data.empty:

        st.warning(
            "Risk-scored transaction data not found."
        )

    else:

        total = len(risk_data)

        low = (
            risk_data["risk_level"]
            .eq("LOW")
            .sum()
        )

        medium = (
            risk_data["risk_level"]
            .eq("MEDIUM")
            .sum()
        )

        high = (
            risk_data["risk_level"]
            .eq("HIGH")
            .sum()
        )

        critical = (
            risk_data["risk_level"]
            .eq("CRITICAL")
            .sum()
        )


        c1, c2, c3, c4, c5 = st.columns(5)


        c1.metric(
            "Transactions",
            f"{total:,}",
        )

        c2.metric(
            "Low Risk",
            f"{low:,}",
        )

        c3.metric(
            "Medium",
            f"{medium:,}",
        )

        c4.metric(
            "High",
            f"{high:,}",
        )

        c5.metric(
            "Critical",
            f"{critical:,}",
        )


        st.divider()


        left, right = st.columns(2)


        with left:

            st.subheader(
                "Risk Distribution"
            )

            distribution = pd.DataFrame(
                {
                    "Risk Level": [
                        "LOW",
                        "MEDIUM",
                        "HIGH",
                        "CRITICAL",
                    ],

                    "Transactions": [
                        low,
                        medium,
                        high,
                        critical,
                    ],
                }
            )

            st.bar_chart(
                distribution.set_index(
                    "Risk Level"
                )
            )


        with right:

            st.subheader(
                "Risk Score Statistics"
            )

            st.write(
                risk_data[
                    "risk_score"
                ].describe()
            )


        st.divider()


        st.subheader(
            "🚨 Highest Risk Transactions"
        )


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


        available = [
            c
            for c in display_columns
            if c in risk_data.columns
        ]


        highest_risk = (
            risk_data
            .sort_values(
                "risk_score",
                ascending=False
            )
            .head(20)
        )


        st.dataframe(
            highest_risk[available],
            use_container_width=True,
        )



# ============================================================
# FRAUD ALERT CENTER
# ============================================================

elif page == "🚨 Fraud Alert Center":

    st.header("🚨 Fraud Alert Center")
    st.caption(
        "Real-time prioritization of transactions requiring defensive review"
    )

    if risk_data.empty:

        st.warning("No risk-scored transaction data available.")

    else:

        # ----------------------------------------------------
        # ALERT CLASSIFICATION
        # ----------------------------------------------------

        alert_data = risk_data.copy()

        # Safely create alert priority
        if "risk_level" in alert_data.columns:
            alert_data["alert_priority"] = alert_data["risk_level"].map({
                "CRITICAL": 4,
                "HIGH": 3,
                "MEDIUM": 2,
                "LOW": 1
            }).fillna(0)
        else:
            alert_data["alert_priority"] = 0

        # ----------------------------------------------------
        # TOP ALERT METRICS
        # ----------------------------------------------------

        total_alerts = len(
            alert_data[
                alert_data["risk_level"].isin(["HIGH", "CRITICAL"])
            ]
        )

        critical_alerts = len(
            alert_data[
                alert_data["risk_level"] == "CRITICAL"
            ]
        )

        high_alerts = len(
            alert_data[
                alert_data["risk_level"] == "HIGH"
            ]
        )

        verify_alerts = len(
            alert_data[
                alert_data["recommended_action"]
                .astype(str)
                .str.upper()
                .str.contains("VERIFY")
            ]
        ) if "recommended_action" in alert_data.columns else 0

        hold_alerts = len(
            alert_data[
                alert_data["recommended_action"]
                .astype(str)
                .str.upper()
                .str.contains("HOLD|BLOCK", regex=True)
            ]
        ) if "recommended_action" in alert_data.columns else 0

        # ----------------------------------------------------
        # METRIC CARDS
        # ----------------------------------------------------

        a1, a2, a3, a4, a5 = st.columns(5)

        a1.metric(
            "🚨 Active Alerts",
            f"{total_alerts:,}"
        )

        a2.metric(
            "🔴 Critical",
            f"{critical_alerts:,}"
        )

        a3.metric(
            "🟠 High Risk",
            f"{high_alerts:,}"
        )

        a4.metric(
            "🛡️ Verification",
            f"{verify_alerts:,}"
        )

        a5.metric(
            "⛔ Hold / Block",
            f"{hold_alerts:,}"
        )

        st.divider()

        # ----------------------------------------------------
        # ALERT FILTERS
        # ----------------------------------------------------

        st.subheader("🔎 Alert Investigation")

        f1, f2, f3 = st.columns(3)

        with f1:
            selected_level = st.multiselect(
                "Risk Level",
                ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                default=["CRITICAL", "HIGH"]
            )

        with f2:
            selected_action = st.multiselect(
                "Recommended Action",
                sorted(
                    alert_data["recommended_action"]
                    .dropna()
                    .astype(str)
                    .unique()
                ) if "recommended_action" in alert_data.columns
                else [],
                default=[]
            )

        with f3:
            minimum_score = st.slider(
                "Minimum Risk Score",
                min_value=0,
                max_value=100,
                value=60
            )

        # ----------------------------------------------------
        # APPLY FILTERS
        # ----------------------------------------------------

        filtered = alert_data.copy()

        if selected_level:
            filtered = filtered[
                filtered["risk_level"].isin(selected_level)
            ]

        if "risk_score" in filtered.columns:
            filtered = filtered[
                filtered["risk_score"] >= minimum_score
            ]

        if selected_action and "recommended_action" in filtered.columns:
            filtered = filtered[
                filtered["recommended_action"]
                .astype(str)
                .isin(selected_action)
            ]

        # Highest priority first
        filtered = filtered.sort_values(
            ["alert_priority", "risk_score"],
            ascending=[False, False]
        )

        st.write(
            f"Showing **{len(filtered):,}** prioritized transaction(s)"
        )

        # ----------------------------------------------------
        # ALERT TABLE
        # ----------------------------------------------------

        alert_columns = [
            "step",
            "type",
            "amount",
            "ml_probability_percent",
            "behavior_score",
            "risk_score",
            "risk_level",
            "recommended_action"
        ]

        available_alert_columns = [
            c for c in alert_columns
            if c in filtered.columns
        ]

        if not filtered.empty and available_alert_columns:

            st.dataframe(
                filtered[available_alert_columns].head(50),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No transactions match the selected alert criteria."
            )

        # ----------------------------------------------------
        # PRIORITY VISUALIZATION
        # ----------------------------------------------------

        st.divider()

        left, right = st.columns(2)

        with left:

            st.subheader("📊 Alert Severity")

            severity_counts = (
                alert_data["risk_level"]
                .value_counts()
                .reindex(
                    ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                    fill_value=0
                )
            )

            st.bar_chart(severity_counts)

        with right:

            st.subheader("🎯 Recommended Response")

            if "recommended_action" in alert_data.columns:

                action_counts = (
                    alert_data["recommended_action"]
                    .astype(str)
                    .value_counts()
                )

                st.bar_chart(action_counts)

        # ----------------------------------------------------
        # INVESTIGATE HIGHEST-RISK TRANSACTION
        # ----------------------------------------------------

        st.divider()

        st.subheader("🔬 Highest-Priority Alert")

        if not filtered.empty:

            top_alert = filtered.iloc[0]

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Risk Score",
                    f"{float(top_alert.get('risk_score', 0)):.1f}/100"
                )

            with c2:
                st.metric(
                    "Risk Level",
                    str(top_alert.get("risk_level", "UNKNOWN"))
                )

            with c3:
                st.metric(
                    "Action",
                    str(top_alert.get(
                        "recommended_action",
                        "REVIEW"
                    ))
                )

            st.markdown("### 🧠 Decision Evidence")

            evidence = []

            if "amount" in top_alert:
                evidence.append(
                    f"💰 Transaction amount: ₹{float(top_alert['amount']):,.2f}"
                )

            if "ml_probability_percent" in top_alert:
                evidence.append(
                    f"🤖 Model fraud probability: "
                    f"{float(top_alert['ml_probability_percent']):.1f}%"
                )

            if "behavior_score" in top_alert:
                evidence.append(
                    f"🧩 Behavioral risk score: "
                    f"{float(top_alert['behavior_score']):.1f}"
                )

            if "risk_score" in top_alert:
                evidence.append(
                    f"📈 Composite risk score: "
                    f"{float(top_alert['risk_score']):.1f}/100"
                )

            for item in evidence:
                st.write(item)

            # Defensive explanation
            action = str(
                top_alert.get(
                    "recommended_action",
                    "REVIEW"
                )
            ).upper()

            if "BLOCK" in action or "HOLD" in action:

                st.error(
                    "⛔ High-priority defensive response recommended. "
                    "The transaction should be held for additional review "
                    "before completion."
                )

            elif "VERIFY" in action:

                st.warning(
                    "🛡️ Step-up verification recommended. "
                    "The transaction shows elevated risk, but additional "
                    "verification can reduce uncertainty without immediately "
                    "blocking a potentially genuine customer."
                )

            else:

                st.info(
                    "🔎 Review recommended. The transaction has elevated risk "
                    "signals and should be assessed according to the recommended "
                    "security action."
               )



# ============================================================
# PAGE 4 — WHAT-IF SIMULATOR
# ============================================================

elif page == "🧪 What-If Simulator":

    st.header("🧪 What-If Fraud Simulator")

    st.write(
        "Explore how changing transaction behavior can change "
        "FrictionGuard's fraud decision."
    )

    st.info(
        "💡 Change one or more transaction conditions and compare "
        "the original risk with the simulated scenario."
    )

    st.divider()

    # --------------------------------------------------------
    # BASELINE TRANSACTION
    # --------------------------------------------------------

    st.subheader("🎯 Baseline Transaction")

    b1, b2, b3 = st.columns(3)

    with b1:

        base_type = st.selectbox(
            "Transaction Type",
            [
                "PAYMENT",
                "TRANSFER",
                "CASH_OUT",
                "CASH_IN",
                "DEBIT",
            ],
            key="whatif_type"
        )

        base_amount = st.number_input(
            "Transaction Amount",
            min_value=0.01,
            value=10000.0,
            step=1000.0,
            key="whatif_amount"
        )

        base_step = st.number_input(
            "Transaction Step",
            min_value=1,
            value=1,
            key="whatif_step"
        )

    with b2:

        base_sender_txn = st.number_input(
            "Sender Previous Transactions",
            min_value=0,
            value=5,
            key="whatif_sender_txn"
        )

        base_sender_avg = st.number_input(
            "Sender Historical Avg Amount",
            min_value=0.01,
            value=10000.0,
            key="whatif_sender_avg"
        )

        base_sender_recent = st.number_input(
            "Sender Recent Transactions",
            min_value=0,
            value=2,
            key="whatif_sender_recent"
        )

        base_sender_recent_amount = st.number_input(
            "Sender Recent Amount Total",
            min_value=0.0,
            value=20000.0,
            step=1000.0,
            key="whatif_sender_recent_amount",
            help="Sum of the sender's previous recent transaction amounts.",
        )

    with b3:

        base_receiver_txn = st.number_input(
            "Receiver Previous Transactions",
            min_value=0,
            value=5,
            key="whatif_receiver_txn"
        )

        base_receiver_avg = st.number_input(
            "Receiver Historical Avg Amount",
            min_value=0.01,
            value=10000.0,
            key="whatif_receiver_avg"
        )

        base_receiver_recent = st.number_input(
            "Receiver Recent Transactions",
            min_value=0,
            value=2,
            key="whatif_receiver_recent"
        )

        base_receiver_recent_amount = st.number_input(
            "Receiver Recent Amount Total",
            min_value=0.0,
            value=20000.0,
            step=1000.0,
            key="whatif_receiver_recent_amount",
            help="Sum of the receiver's previous recent transaction amounts.",
        )

    b4, b5 = st.columns(2)

    with b4:

        base_sender_balance = st.number_input(
            "Sender Balance",
            min_value=0.0,
            value=25000.0,
            step=1000.0,
            key="whatif_sender_balance"
        )

    with b5:

        base_receiver_balance = st.number_input(
            "Receiver Balance",
            min_value=0.0,
            value=10000.0,
            step=1000.0,
            key="whatif_receiver_balance"
        )

    st.divider()

    # --------------------------------------------------------
    # SCENARIO MODIFIER
    # --------------------------------------------------------

    st.subheader("⚡ Create a What-If Scenario")

    scenario = st.selectbox(
        "Choose the factor you want to change",
        [
            "Transaction Amount",
            "Sender Recent Activity",
            "Receiver Recent Activity",
            "Sender Balance",
            "Receiver Balance",
            "Sender Historical Average",
            "Receiver Historical Average",
        ],
        key="whatif_factor"
    )

    st.caption(
        "The baseline remains unchanged. Only the selected factor "
        "will be modified in the simulated transaction."
    )

    # Default scenario values
    scenario_amount = base_amount
    scenario_sender_recent = base_sender_recent
    scenario_receiver_recent = base_receiver_recent
    scenario_sender_recent_amount = base_sender_recent_amount
    scenario_receiver_recent_amount = base_receiver_recent_amount
    scenario_sender_balance = base_sender_balance
    scenario_receiver_balance = base_receiver_balance
    scenario_sender_avg = base_sender_avg
    scenario_receiver_avg = base_receiver_avg

    if scenario == "Transaction Amount":

        scenario_amount = st.slider(
            "Simulated Transaction Amount",
            min_value=100.0,
            max_value=1000000.0,
            value=float(base_amount),
            step=1000.0,
            key="scenario_amount"
        )

    elif scenario == "Sender Recent Activity":

        scenario_sender_recent = st.slider(
            "Simulated Sender Recent Transactions",
            min_value=0,
            max_value=30,
            value=int(base_sender_recent),
            key="scenario_sender_recent"
        )

        scenario_sender_recent_amount = st.number_input(
            "Simulated Sender Recent Amount Total",
            min_value=0.0,
            value=float(base_sender_recent_amount),
            step=1000.0,
            key="scenario_sender_recent_amount",
        )

    elif scenario == "Receiver Recent Activity":

        scenario_receiver_recent = st.slider(
            "Simulated Receiver Recent Transactions",
            min_value=0,
            max_value=30,
            value=int(base_receiver_recent),
            key="scenario_receiver_recent"
        )

        scenario_receiver_recent_amount = st.number_input(
            "Simulated Receiver Recent Amount Total",
            min_value=0.0,
            value=float(base_receiver_recent_amount),
            step=1000.0,
            key="scenario_receiver_recent_amount",
        )

    elif scenario == "Sender Balance":

        scenario_sender_balance = st.slider(
            "Simulated Sender Balance",
            min_value=0.0,
            max_value=1000000.0,
            value=float(base_sender_balance),
            step=1000.0,
            key="scenario_sender_balance"
        )

    elif scenario == "Receiver Balance":

        scenario_receiver_balance = st.slider(
            "Simulated Receiver Balance",
            min_value=0.0,
            max_value=1000000.0,
            value=float(base_receiver_balance),
            step=1000.0,
            key="scenario_receiver_balance"
        )

    elif scenario == "Sender Historical Average":

        scenario_sender_avg = st.slider(
            "Simulated Sender Historical Average",
            min_value=100.0,
            max_value=1000000.0,
            value=float(base_sender_avg),
            step=1000.0,
            key="scenario_sender_avg"
        )

    elif scenario == "Receiver Historical Average":

        scenario_receiver_avg = st.slider(
            "Simulated Receiver Historical Average",
            min_value=100.0,
            max_value=1000000.0,
            value=float(base_receiver_avg),
            step=1000.0,
            key="scenario_receiver_avg"
        )

    st.divider()

    # --------------------------------------------------------
    # RUN SIMULATION
    # --------------------------------------------------------

    simulate = st.button(
        "🚀 RUN WHAT-IF SIMULATION",
        use_container_width=True
    )

    if simulate:

        if model is None:

            st.error("Fraud detection model not found.")

        else:

            # -----------------------------------------------
            # BASELINE ANALYSIS
            # -----------------------------------------------

            baseline = analyze_transaction(
                base_step,
                base_type,
                base_amount,
                base_sender_txn,
                base_sender_avg,
                base_receiver_txn,
                base_receiver_avg,
                base_sender_recent,
                base_receiver_recent,
                base_sender_recent_amount,
                base_receiver_recent_amount,
                base_sender_balance,
                base_receiver_balance,
            )

            # -----------------------------------------------
            # SIMULATED ANALYSIS
            # -----------------------------------------------

            simulated = analyze_transaction(
                base_step,
                base_type,
                scenario_amount,
                base_sender_txn,
                scenario_sender_avg,
                base_receiver_txn,
                scenario_receiver_avg,
                scenario_sender_recent,
                scenario_receiver_recent,
                scenario_sender_recent_amount,
                scenario_receiver_recent_amount,
                scenario_sender_balance,
                scenario_receiver_balance,
            )

            st.divider()

            st.subheader("📊 Risk Impact Analysis")

            # -----------------------------------------------
            # METRIC COMPARISON
            # -----------------------------------------------

            c1, c2, c3 = st.columns(3)

            risk_change = (
                simulated["risk_score"]
                - baseline["risk_score"]
            )

            probability_change = (
                simulated["probability"]
                - baseline["probability"]
            )

            behavior_change = (
                simulated["behavioral_score"]
                - baseline["behavioral_score"]
            )

            with c1:

                st.metric(
                    "Risk Score",
                    f"{simulated['risk_score']:.1f}/100",
                    f"{risk_change:+.1f}"
                )

            with c2:

                st.metric(
                    "ML Fraud Probability",
                    f"{simulated['probability'] * 100:.2f}%",
                    f"{probability_change * 100:+.2f}%"
                )

            with c3:

                st.metric(
                    "Behavior Score",
                    f"{simulated['behavioral_score']}/83",
                    f"{behavior_change:+.0f}"
                )

            st.divider()

            # -----------------------------------------------
            # BEFORE / AFTER
            # -----------------------------------------------

            st.subheader("🔄 Before vs After")

            comparison = pd.DataFrame(
                {
                    "Metric": [
                        "Risk Score",
                        "ML Fraud Probability",
                        "Behavior Score",
                        "Risk Level",
                        "Recommended Action",
                    ],
                    "Baseline": [
                        f"{baseline['risk_score']:.1f}/100",
                        f"{baseline['probability'] * 100:.2f}%",
                        f"{baseline['behavioral_score']}/83",
                        baseline["level"],
                        baseline["action"],
                    ],
                    "What-If Scenario": [
                        f"{simulated['risk_score']:.1f}/100",
                        f"{simulated['probability'] * 100:.2f}%",
                        f"{simulated['behavioral_score']}/83",
                        simulated["level"],
                        simulated["action"],
                    ],
                }
            )

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # -----------------------------------------------
            # RISK TRANSITION
            # -----------------------------------------------

            st.subheader("🚦 Risk Transition")

            r1, r2, r3 = st.columns(3)

            with r1:

                st.markdown("### 🟦 BEFORE")

                st.metric(
                    "Risk",
                    baseline["level"]
                )

                st.progress(
                    min(
                        baseline["risk_score"] / 100,
                        1.0
                    )
                )

            with r2:

                st.markdown("### ➡️ CHANGE")

                if risk_change > 0:

                    st.error(
                        f"Risk increased by "
                        f"{risk_change:+.1f} points"
                    )

                elif risk_change < 0:

                    st.success(
                        f"Risk decreased by "
                        f"{risk_change:+.1f} points"
                    )

                else:

                    st.info(
                        "Risk score remained unchanged."
                    )

            with r3:

                st.markdown("### 🟥 AFTER")

                st.metric(
                    "Risk",
                    simulated["level"]
                )

                st.progress(
                    min(
                        simulated["risk_score"] / 100,
                        1.0
                    )
                )

            st.divider()

            # -----------------------------------------------
            # EXPLAIN WHY
            # -----------------------------------------------

            
            if baseline["risk_score"] == simulated["risk_score"]:
                st.subheader("🧠 Why Did the Decision Stay the Same?")
            else:
                st.subheader("🧠 Why Did the Decision Change?")

            explanation_found = False

            # Compare transaction amount
            if baseline["risk_score"] != simulated["risk_score"]:

                amount_change = scenario_amount - base_amount

                if amount_change > 0:
                    st.warning(
                        f"💰 Transaction amount increased from "
                        f"₹{base_amount:,.2f} to ₹{scenario_amount:,.2f}."
                    )
                    explanation_found = True

                elif amount_change < 0:
                    st.success(
                        f"💰 Transaction amount decreased from "
                        f"₹{base_amount:,.2f} to ₹{scenario_amount:,.2f}."
                    )
                    explanation_found = True


            # Compare behavioral score
            behavior_change = (
                simulated["behavioral_score"]
                - baseline["behavioral_score"]
            )

            if behavior_change > 0:
                st.warning(
                    f"🧩 Behavioral risk increased by "
                    f"{behavior_change:.1f} points "
                    f"({baseline['behavioral_score']:.1f} → "
                    f"{simulated['behavioral_score']:.1f})."
                )
                explanation_found = True

            elif behavior_change < 0:
                st.success(
                    f"🧩 Behavioral risk decreased by "
                    f"{abs(behavior_change):.1f} points "
                    f"({baseline['behavioral_score']:.1f} → "
                    f"{simulated['behavioral_score']:.1f})."
                )
                explanation_found = True


            # Show actual model / behavioral reasons
            if simulated["reasons"]:

                for reason in simulated["reasons"]:

                    # Don't display the generic message when
                    # we already have meaningful comparison results
                    if reason != "No significant anomalous behavior detected.":
                        st.markdown(
                            f"""
                            <div class="reason">
                            ⚠️ {reason}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


            # Compare overall risk
            risk_change = (
                simulated["risk_score"]
                - baseline["risk_score"]
            )

            if risk_change > 0:

                st.warning(
                    f"📈 Overall risk increased by "
                    f"{risk_change:.1f} points "
                    f"({baseline['risk_score']:.1f} → "
                    f"{simulated['risk_score']:.1f})."
                )
                explanation_found = True

            elif risk_change < 0:

                st.success(
                    f"📉 Overall risk decreased by "
                    f"{abs(risk_change):.1f} points "
                    f"({baseline['risk_score']:.1f} → "
                    f"{simulated['risk_score']:.1f})."
                )
                explanation_found = True


            if not explanation_found:

                st.info(
                    "The simulated transaction produced no measurable "
                    "change in the evaluated risk signals."
                )

            st.divider()
            
            
            

            # -----------------------------------------------
            # FINAL DECISION
            # -----------------------------------------------

            st.subheader("🎯 Simulated Security Decision")

            if simulated["level"] == "CRITICAL":

                st.error(
                    f"🚨 CRITICAL — {simulated['action']}"
                )

            elif simulated["level"] == "HIGH":

                st.warning(
                    f"🟠 HIGH RISK — {simulated['action']}"
                )

            elif simulated["level"] == "MEDIUM":

                st.info(
                    f"🟡 MEDIUM RISK — {simulated['action']}"
                )

            else:

                st.success(
                    f"🟢 LOW RISK — {simulated['action']}"
                )



# ============================================================
# PAGE 5 — SYSTEM INFORMATION
# ============================================================

elif page == "ℹ️ System Information":

    st.header("ℹ️ FrictionGuard AI")

    st.markdown(
        """
        ## System Architecture

        **FrictionGuard AI** is an explainable AI
        fraud detection platform.

        ### Detection Pipeline

        1. Raw financial transactions
        2. Data quality analysis
        3. Leakage protection
        4. Behavioral feature engineering
        5. Machine learning
        6. Fraud probability estimation
        7. Behavioral risk analysis
        8. Composite risk scoring
        9. Explainable fraud reasons
        10. Recommended security action

        ### Machine Learning

        The current system evaluates:

        - Logistic Regression
        - Random Forest

        The Random Forest model was selected using
        **PR-AUC**, which is particularly useful for
        highly imbalanced fraud detection.

        ### Risk Intelligence

        FrictionGuard combines:

        **ML Probability + Behavioral Signals**

        to produce a unified:

        **0–100 Risk Score**

        ### Decision Levels

        | Risk | Action |
        |---|---|
        | LOW | Allow |
        | MEDIUM | Additional Verification |
        | HIGH | Step-up Authentication / Review |
        | CRITICAL | Block / Manual Investigation |

        ### Explainability

        Instead of returning only:

        `Fraud = 1`

        the system explains *why* the transaction
        was considered suspicious.

        This makes the system more suitable for
        real-world financial risk workflows.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FrictionGuard AI • Explainable Fraud Intelligence Platform"
)