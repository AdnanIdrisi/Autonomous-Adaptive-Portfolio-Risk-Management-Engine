# 📊 Regime-Based Portfolio Allocation with Risk Engine & Stress Testing

This project implements a **Hybrid Regime-Based Portfolio Allocation System** using:

- Hidden Markov Model (Regime Detection)
- Dynamic Asset Allocation Engine
- Risk Management Module
- Rolling Window Backtesting
- Stress Testing (Market Crash & Volatility Spike)

It simulates how a portfolio behaves under different market regimes and evaluates robustness using stress scenarios.

---

## 🚀 Features
- Regime detection using rolling window training
- Hybrid allocation based on volatility + market regime
- Risk engine with weight clipping & normalization
- Rolling backtest engine
- Stress testing module:
  - Equity Market Crash
  - Volatility Spike
- Portfolio performance metrics (Sharpe, Sortino, Drawdown, etc.)

---

## 📁 Project Structure
```
project-root/
│
├── backend/
│ ├── core/
│ │ ├── data_loader.py
│ │ ├── feature_engineering.py
│ │ ├── regime_detection.py
│ │ ├── allocation_engine.py
│ │ ├── risk_manager.py
│ │ ├── stress_testing.py
│ │
│ └── utils/
│   └── metrics.py
│
├── backtester.py
├── requirements.txt
└── README.md
```
---

## 🛠️ Setup Instructions

### 1️⃣ Create Virtual Environment
```bash
python -m venv .venv
```
Activate the environment:

Windows
```
.venv\Scripts\activate
```

Mac/Linux
```
source .venv/bin/activate
```


### 2️⃣ Install Required Libraries
```
pip install -r requirements.txt
```

### ▶️ Running the Project
Open new terminal
```
uvicorn backend.main:app --reload
```

Open new terminal
```
streamlit run frontend/app.py
```


# 🧠 Methodology
Regime Detection

Hidden Markov Model (HMM) identifies hidden market states using rolling feature windows.

Allocation Engine

Allocates weights dynamically based on:

Current volatility

Predicted market regime

Risk Manager

Removes negative weights (no short selling)

Re-normalizes allocation to sum = 1

Prevents extreme portfolio exposure

## Stress Testing

Simulates:

Market crash (sharp negative shocks)

Volatility spike (increased return dispersion)


# 🎯 Future Improvements

Add transaction costs

Include more asset classes

Integrate live data API

Deploy Streamlit dashboard