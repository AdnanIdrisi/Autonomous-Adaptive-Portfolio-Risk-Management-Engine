import pandas as pd
import numpy as np

from data_loader import fetch_price_data
from feature_engineering import (
    build_feature_set,
    compute_returns,
    compute_rolling_volatility
)
from regime_detection import RegimeDetector
from allocation_engine import AllocationEngine
from risk_manager import RiskManager


class Backtester:
    def __init__(self, window: int = 120):
        self.window = window

    def run_backtest(self):
        """
        Perform rolling window backtest.
        """
        # Load data
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

        for i in range(self.window, len(common_index) - 1):
            train_slice = slice(i - self.window, i)
            test_date = common_index[i]

            # Training data
            train_features = features.iloc[train_slice]
            train_vol = vol.iloc[train_slice]

            # Fit regime detector
            detector = RegimeDetector()
            detector.fit(train_features)

            # Predict regime for test day
            test_feature = features.iloc[[i]]
            regime = detector.predict(test_feature).iloc[0]

            # Compute allocation using last known vol
            current_vol = vol.iloc[[i]]
            weights = alloc_engine.compute_allocation(
                current_vol,
                pd.Series([regime], index=current_vol.index)
            )

            # Apply risk manager
            recent_returns = returns.iloc[:i+1]
            adjusted_weights = risk_manager.apply_risk_controls(weights, recent_returns)

            # --- SAFETY FIX ---
            w = adjusted_weights.iloc[0].copy()

            # Remove negative weights (no short selling)
            w = w.clip(lower=0)

            # Re-normalize to sum = 1
            if w.sum() == 0:
                w = pd.Series([1/len(w)] * len(w), index=w.index)
            else:
                w = w / w.sum()

            next_date = common_index[i+1]

            # print("\nRaw weights:", adjusted_weights.iloc[0])
            # print("Clipped weights:", w)
            # print("Sum weights:", w.sum())
            # print("Next day returns:", returns.iloc[i+1])


            # Compute next day return
            next_return = (w * returns.loc[next_date]).sum()

            # 🔴 Catch abnormal returns
            if next_return < -0.5 or next_return > 0.5:
                print("\n🚨 EXTREME RETURN DETECTED 🚨")
                print("Date:", common_index[i+1])
                print("Weights used:\n", w)
                print("Next day returns:\n", returns.iloc[i+1])
                print("Computed return:", next_return)
                raise ValueError("Extreme return detected! Check this date.")



            portfolio_returns.append(next_return)
            dates.append(common_index[i+1])

        portfolio_returns = pd.Series(
            portfolio_returns,
            index=dates,
            name="Portfolio_Return"
        )
        return portfolio_returns


if __name__ == "__main__":
    backtester = Backtester(window=120)
    port_returns = backtester.run_backtest()

    print(port_returns.head())
    print("Min daily return:", port_returns.min())
    print("\nTotal Return:", (1 + port_returns).prod() - 1)

    # Performance Metrics
    cumulative = (1 + port_returns).cumprod()
    annual_return = cumulative.iloc[-1] ** (252 / len(port_returns)) - 1
    annual_vol = port_returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol

    print("\nAnnual Return:", annual_return)
    print("Annual Volatility:", annual_vol)
    print("Sharpe Ratio:", sharpe)

