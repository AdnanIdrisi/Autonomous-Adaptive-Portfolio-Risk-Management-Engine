import pandas as pd
import numpy as np


class RiskManager:
    def __init__(
        self,
        target_vol: float = 0.15,      # 15% annualized vol target
        max_drawdown: float = -0.20,   # -20% drawdown threshold
        stop_loss: float = -0.05       # -5% recent loss trigger
    ):
        self.target_vol = target_vol
        self.max_drawdown = max_drawdown
        self.stop_loss = stop_loss

    def compute_portfolio_volatility(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        window: int = 20
    ) -> pd.Series:
        """
        Compute rolling portfolio volatility.
        """
        port_returns = (weights * returns).sum(axis=1)
        port_vol = port_returns.rolling(window).std() * np.sqrt(252)
        return port_vol

    def compute_portfolio_drawdown(self, portfolio_returns: pd.Series) -> pd.Series:
        """
        Compute portfolio drawdown over time.
        """
        cumulative = (1 + portfolio_returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown

    def apply_risk_controls(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Adjust weights dynamically based on risk rules.
        """
        adjusted_weights = weights.copy()
        port_returns = (weights * returns).sum(axis=1)

        # Compute risk metrics
        port_vol = self.compute_portfolio_volatility(weights, returns)
        drawdown = self.compute_portfolio_drawdown(port_returns)

        for date in weights.index:
            if date not in port_vol.index:
                continue

            # Volatility targeting
            if port_vol.loc[date] > self.target_vol:
                adjusted_weights.loc[date] *= 0.7  # reduce exposure

            # Drawdown protection
            if drawdown.loc[date] < self.max_drawdown:
                adjusted_weights.loc[date] *= 0.5  # aggressive risk cut

            # Stop-loss logic (recent negative return)
            if port_returns.loc[date] < self.stop_loss:
                adjusted_weights.loc[date] *= 0.6

            # Re-normalize weights
            adjusted_weights.loc[date] = (
                adjusted_weights.loc[date] /
                adjusted_weights.loc[date].sum()
            )

        return adjusted_weights


if __name__ == "__main__":
    # from backend.core.data_loader import fetch_price_data
    from data_loader import fetch_price_data
    # from backend.core.feature_engineering import compute_returns, compute_rolling_volatility
    from feature_engineering import compute_returns, compute_rolling_volatility
    # from backend.core.regime_detection import RegimeDetector
    from regime_detection import RegimeDetector
    # from backend.core.allocation_engine import AllocationEngine
    from allocation_engine import AllocationEngine
    # from backend.core.feature_engineering import build_feature_set
    from feature_engineering import build_feature_set

    prices = fetch_price_data()
    returns = compute_returns(prices)
    vol = compute_rolling_volatility(returns)
    features = build_feature_set(prices)

    # Align indices
    common_index = vol.index.intersection(features.index).intersection(returns.index)
    vol = vol.loc[common_index]
    features = features.loc[common_index]
    returns = returns.loc[common_index]

    # Regime Detection
    detector = RegimeDetector()
    regimes = detector.fit_predict(features)

    # Allocation
    alloc_engine = AllocationEngine()
    weights = alloc_engine.compute_allocation(vol, regimes)

    # Risk Manager
    risk_manager = RiskManager()
    adjusted_weights = risk_manager.apply_risk_controls(weights, returns)

    print(adjusted_weights.head())
    print("\nAdjusted Weights Sum:")
    print(adjusted_weights.sum(axis=1).tail())
