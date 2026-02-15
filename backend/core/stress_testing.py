import pandas as pd
import numpy as np


class StressTester:
    """
    Stress testing engine to evaluate portfolio robustness
    under extreme market scenarios.
    """

    def __init__(self, crash_magnitude: float = -0.2, vol_multiplier: float = 2.0):
        """
        Parameters:
        crash_magnitude : percentage crash applied to equities
        vol_multiplier  : factor to increase volatility
        """
        self.crash_magnitude = crash_magnitude
        self.vol_multiplier = vol_multiplier

    # ------------------------------------------------------------------
    # 1️⃣ Equity Market Crash Scenario
    # ------------------------------------------------------------------
    def simulate_market_crash(self, asset_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a sudden crash to equity assets on worst historical day.
        """
        stressed = asset_returns.copy()

        # Identify worst portfolio day historically
        crash_day = asset_returns.sum(axis=1).idxmin()

        # Apply crash only to equity assets
        stressed.loc[crash_day, ["NIFTY", "BANKNIFTY"]] += self.crash_magnitude

        return stressed

    # ------------------------------------------------------------------
    # 2️⃣ Volatility Spike Scenario
    # ------------------------------------------------------------------
    def simulate_volatility_spike(self, asset_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Increase volatility artificially by scaling returns.
        """
        stressed_returns = asset_returns.copy() * self.vol_multiplier
        return stressed_returns

    # ------------------------------------------------------------------
    # 3️⃣ Regime Shock Scenario (optional future use)
    # ------------------------------------------------------------------
    def simulate_regime_misclassification(self, regimes: pd.Series) -> pd.Series:
        """
        Flip regimes randomly to simulate model mistakes.
        """
        shocked_regimes = regimes.copy()
        flip_mask = np.random.rand(len(regimes)) < 0.2  # 20% wrong predictions
        shocked_regimes[flip_mask] = 1 - shocked_regimes[flip_mask]
        return shocked_regimes

    # ------------------------------------------------------------------
    # Utility: Max Drawdown
    # ------------------------------------------------------------------
    def _max_drawdown(self, returns: pd.Series) -> float:
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    # ------------------------------------------------------------------
    # Compute portfolio returns from asset returns + weights
    # ------------------------------------------------------------------
    def _compute_portfolio_returns(
        self,
        asset_returns: pd.DataFrame,
        weights: pd.DataFrame
    ) -> pd.Series:
        """
        Convert asset-level stressed returns into portfolio returns.
        """
        # Align indices
        common_idx = asset_returns.index.intersection(weights.index)
        asset_returns = asset_returns.loc[common_idx]
        weights = weights.loc[common_idx]

        port_returns = (asset_returns * weights).sum(axis=1)
        return port_returns

    # ------------------------------------------------------------------
    # Run All Stress Scenarios (REQUIRED BY PIPELINE)
    # ------------------------------------------------------------------
    def run_all_scenarios(
        self,
        asset_returns: pd.DataFrame,
        weights: pd.DataFrame,
        portfolio_returns: pd.Series
    ) -> dict:
        """
        Execute all stress scenarios and compare results.
        """
        results = {}

        # Scenario 1: Equity Crash
        crash_asset_returns = self.simulate_market_crash(asset_returns)
        crash_port_returns = self._compute_portfolio_returns(crash_asset_returns, weights)

        results["Equity Market Crash"] = {
            "Normal Total Return": (1 + portfolio_returns).prod() - 1,
            "Stressed Total Return": (1 + crash_port_returns).prod() - 1,
            "Max Drawdown (Stress)": self._max_drawdown(crash_port_returns),
        }

        # Scenario 2: Volatility Spike
        vol_asset_returns = self.simulate_volatility_spike(asset_returns)
        vol_port_returns = self._compute_portfolio_returns(vol_asset_returns, weights)

        results["Volatility Spike"] = {
            "Normal Total Return": (1 + portfolio_returns).prod() - 1,
            "Stressed Total Return": (1 + vol_port_returns).prod() - 1,
            "Max Drawdown (Stress)": self._max_drawdown(vol_port_returns),
        }

        return results


# ------------------------------------------------------------------
# Debug Run
# ------------------------------------------------------------------
if __name__ == "__main__":
    from backend.core.backtester import Backtester
    from backend.core.feature_engineering import compute_returns
    from backend.core.data_loader import fetch_price_data

    # Generate base portfolio
    backtester = Backtester(window=120)
    port_returns = backtester.run_backtest()

    # Need asset returns + weights for stress scenarios
    prices = fetch_price_data()
    asset_returns = compute_returns(prices)

    # Dummy equal weights for debug test
    weights = pd.DataFrame(
        np.repeat([[1/3, 1/3, 1/3]], len(asset_returns), axis=0),
        index=asset_returns.index,
        columns=["NIFTY", "BANKNIFTY", "GOLD"]
    )

    tester = StressTester()
    stress_results = tester.run_all_scenarios(asset_returns, weights, port_returns)

    print("\n===== Stress Test Results =====")
    for scenario, res in stress_results.items():
        print(f"\nScenario: {scenario}")
        for k, v in res.items():
            print(f"{k}: {v:.4f}")
