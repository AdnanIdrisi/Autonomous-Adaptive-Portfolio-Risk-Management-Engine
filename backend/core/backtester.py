import pandas as pd
import numpy as np

from backend.core.data_loader import fetch_price_data
from backend.core.feature_engineering import (
    build_feature_set,
    compute_returns,
    compute_rolling_volatility
)
from backend.core.regime_detection import RegimeDetector
from backend.core.allocation_engine import AllocationEngine
from backend.core.risk_manager import RiskManager


class Backtester:
    def __init__(self, window: int = 120, use_risk: bool = True):
        self.window = window
        self.use_risk = use_risk  # 🔥 Toggle risk engine on/off

    def run_backtest(self):
        """
        Perform rolling window backtest with optional risk engine.
        """
        # ---------------------------
        # 1️⃣ Load & Prepare Data
        # ---------------------------
        prices = fetch_price_data()
        returns = compute_returns(prices)
        features = build_feature_set(prices)
        vol = compute_rolling_volatility(returns)

        # Align indices
        common_index = (
            returns.index
            .intersection(features.index)
            .intersection(vol.index)
        )

        returns = returns.loc[common_index]
        features = features.loc[common_index]
        vol = vol.loc[common_index]

        portfolio_returns = []
        dates = []

        alloc_engine = AllocationEngine()
        risk_manager = RiskManager()

        # ---------------------------
        # 2️⃣ Rolling Backtest Loop
        # ---------------------------
        for i in range(self.window, len(common_index) - 1):
            train_slice = slice(i - self.window, i)
            next_date = common_index[i + 1]

            # Train regime detector on past data
            train_features = features.iloc[train_slice]
            detector = RegimeDetector()
            detector.fit(train_features)

            # Predict regime for current day
            test_feature = features.iloc[[i]]
            regime = detector.predict(test_feature).iloc[0]

            # Compute allocation based on volatility + regime
            current_vol = vol.iloc[[i]]
            weights = alloc_engine.compute_allocation(
                current_vol,
                pd.Series([regime], index=current_vol.index)
            )

            # ---------------------------
            # 3️⃣ Apply Risk Engine (optional)
            # ---------------------------
            if self.use_risk:
                recent_returns = returns.iloc[:i + 1]
                adjusted_weights = risk_manager.apply_risk_controls(weights, recent_returns)
            else:
                adjusted_weights = weights

            # Safety processing
            w = adjusted_weights.iloc[0].copy()
            w = w.clip(lower=0)  # no short selling

            if w.sum() == 0:
                w = pd.Series([1 / len(w)] * len(w), index=w.index)
            else:
                w = w / w.sum()

            # ---------------------------
            # 4️⃣ Compute Next-Day Return
            # ---------------------------
            next_return = (w * returns.loc[next_date]).sum()

            # Catch abnormal returns
            if next_return < -0.5 or next_return > 0.5:
                print("\n🚨 EXTREME RETURN DETECTED 🚨")
                print("Date:", next_date)
                print("Weights used:\n", w)
                print("Next day returns:\n", returns.loc[next_date])
                print("Computed return:", next_return)
                raise ValueError("Extreme return detected! Check this date.")

            portfolio_returns.append(next_return)
            dates.append(next_date)

        # ---------------------------
        # 5️⃣ Output Series
        # ---------------------------
        portfolio_returns = pd.Series(
            portfolio_returns,
            index=dates,
            name="Portfolio_Return"
        )

        return portfolio_returns


if __name__ == "__main__":
    # With Risk Engine
    bt_with_risk = Backtester(window=120, use_risk=True)
    returns_with_risk = bt_with_risk.run_backtest()

    # Without Risk Engine
    bt_without_risk = Backtester(window=120, use_risk=False)
    returns_without_risk = bt_without_risk.run_backtest()

    print("\n=== Backtest Completed ===")
    print("With Risk Engine Total Return:", (1 + returns_with_risk).prod() - 1)
    print("Without Risk Engine Total Return:", (1 + returns_without_risk).prod() - 1)
