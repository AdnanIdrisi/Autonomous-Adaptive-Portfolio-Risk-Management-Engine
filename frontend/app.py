import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000/run-pipeline"

st.set_page_config(page_title="Adaptive Portfolio Dashboard", layout="wide")

st.title("📊 Adaptive Portfolio & Risk Management Dashboard")

# ---------------------------------------------------
# Fetch Data from FastAPI
# ---------------------------------------------------
@st.cache_data
def fetch_pipeline_data():
    response = requests.get(API_URL)
    return response.json()

data = fetch_pipeline_data()

# ---------------------------------------------------
# Extract Data
# ---------------------------------------------------
performance = data["performance"]
portfolio_returns = pd.Series(data["portfolio_returns"])
regimes = pd.Series(data["regime_history"])
volatility = pd.DataFrame(data["volatility_series"])
weights = data["latest_weights"]
stress = data["stress_test"]

# ---------------------------------------------------
# 1️⃣ Performance Metrics
# ---------------------------------------------------
st.subheader("📈 Portfolio Performance Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Total Return", f"{performance['total_return']:.2%}")
col2.metric("Annual Return", f"{performance['annual_return']:.2%}")
col3.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")

col4, col5, col6 = st.columns(3)
col4.metric("Annual Volatility", f"{performance['annual_volatility']:.2%}")
col5.metric("Sortino Ratio", f"{performance['sortino_ratio']:.2f}")
col6.metric("Max Drawdown", f"{performance['max_drawdown']:.2%}")

st.divider()

# ---------------------------------------------------
# 2️⃣ Equity Curve
# ---------------------------------------------------
st.subheader("📉 Portfolio Equity Curve")

cumulative = (1 + portfolio_returns).cumprod()

fig, ax = plt.subplots()
ax.plot(cumulative)
ax.set_title("Cumulative Portfolio Growth")
ax.set_xlabel("Time")
ax.set_ylabel("Portfolio Value")
st.pyplot(fig)

st.divider()

# ---------------------------------------------------
# 3️⃣ Regime Detection Timeline
# ---------------------------------------------------
st.subheader("🧠 Market Regime Timeline")

fig2, ax2 = plt.subplots()
ax2.plot(regimes, drawstyle="steps-post")
ax2.set_title("Detected Market Regimes")
ax2.set_xlabel("Time")
ax2.set_ylabel("Regime (0=Bear, 1=Bull)")
st.pyplot(fig2)

st.divider()

# ---------------------------------------------------
# 4️⃣ Latest Allocation Weights
# ---------------------------------------------------
st.subheader("🥧 Latest Portfolio Allocation")

fig3, ax3 = plt.subplots()
ax3.pie(weights.values(), labels=weights.keys(), autopct="%1.1f%%")
ax3.set_title("Current Asset Allocation")
st.pyplot(fig3)

st.divider()

# ---------------------------------------------------
# 5️⃣ Stress Test Results
# ---------------------------------------------------
st.subheader("⚠️ Stress Testing Results")

for scenario, results in stress.items():
    st.markdown(f"### {scenario}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Normal Return", f"{results['Normal Total Return']:.2%}")
    col2.metric("Stressed Return", f"{results['Stressed Total Return']:.2%}")
    col3.metric("Max Drawdown", f"{results['Max Drawdown (Stress)']:.2%}")
