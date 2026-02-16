import pandas as pd

from backend.core.data_loader import fetch_price_data
from backend.core.feature_engineering import (
    build_feature_set,
    compute_returns,
    compute_rolling_volatility
)
from backend.core.regime_detection import RegimeDetector
from backend.core.allocation_engine import AllocationEngine
from backend.core.risk_manager import RiskManager
from backend.core.backtester import Backtester
from backend.core.stress_testing import StressTester
from backend.utils.metrics import PortfolioMetrics


class PipelineService:
    """
    Orchestrates the full adaptive portfolio pipeline:
    - Data loading
    - Feature engineering
    - Regime detection
    - Allocation + risk management
    - Backtesting
    - Metrics computation
    - Risk comparison (With vs Without Risk Engine)
    - Stress testing
    """

    def __init__(self, window: int = 120):
        self.window = window

    def _to_python(self, obj):
        """
        Recursively convert numpy & pandas objects to native Python types.
        """
        import numpy as np
        import pandas as pd

        if isinstance(obj, dict):
            return {k: self._to_python(v) for k, v in obj.items()}

        elif isinstance(obj, (list, tuple)):
            return [self._to_python(v) for v in obj]

        elif isinstance(obj, pd.Series):
            return {str(k): self._to_python(v) for k, v in obj.to_dict().items()}

        elif isinstance(obj, pd.DataFrame):
            return {
                str(idx): {col: self._to_python(val) for col, val in row.items()}
                for idx, row in obj.iterrows()
            }

        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()

        elif isinstance(obj, (np.integer,)):
            return int(obj)

        elif isinstance(obj, (np.floating,)):
            return float(obj)

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        return obj

    def run_pipeline(self):
        """
        Run full pipeline and return all artifacts needed
        for API & frontend dashboard.
        """

        # ---------------------------
        # 1️⃣ Load Market Data
        # ---------------------------
        prices = fetch_price_data()

        # ---------------------------
        # 2️⃣ Feature Engineering
        # ---------------------------
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

        # ---------------------------
        # 3️⃣ Regime Detection
        # ---------------------------
        detector = RegimeDetector()
        regimes = detector.fit_predict(features)

        # ---------------------------
        # 4️⃣ Backtesting (WITH Risk Engine)
        # ---------------------------
        backtester = Backtester(window=self.window, use_risk=True)
        portfolio_returns = backtester.run_backtest()

        # ---------------------------
        # 5️⃣ Performance Metrics
        # ---------------------------
        metrics = PortfolioMetrics(portfolio_returns)

        performance_summary = {
            "total_return": metrics.total_return(),
            "annual_return": metrics.annual_return(),
            "annual_volatility": metrics.annual_volatility(),
            "sharpe_ratio": metrics.sharpe_ratio(),
            "sortino_ratio": metrics.sortino_ratio(),
            "max_drawdown": metrics.max_drawdown(),
        }

        # ---------------------------
        # 6️⃣ Risk Engine Comparison
        # ---------------------------
        backtester_no_risk = Backtester(window=self.window, use_risk=False)
        returns_no_risk = backtester_no_risk.run_backtest()

        metrics_no_risk = PortfolioMetrics(returns_no_risk)

        risk_comparison = {
            "Without Risk Engine": {
                "Total Return": metrics_no_risk.total_return(),
                "Annual Return": metrics_no_risk.annual_return(),
                "Annual Volatility": metrics_no_risk.annual_volatility(),
                "Sharpe Ratio": metrics_no_risk.sharpe_ratio(),
                "Sortino Ratio": metrics_no_risk.sortino_ratio(),
                "Max Drawdown": metrics_no_risk.max_drawdown(),
            },
            "With Risk Engine": {
                "Total Return": performance_summary["total_return"],
                "Annual Return": performance_summary["annual_return"],
                "Annual Volatility": performance_summary["annual_volatility"],
                "Sharpe Ratio": performance_summary["sharpe_ratio"],
                "Sortino Ratio": performance_summary["sortino_ratio"],
                "Max Drawdown": performance_summary["max_drawdown"],
            },
        }

        # ---------------------------
        # 7️⃣ Latest Allocation Weights
        # ---------------------------
        alloc_engine = AllocationEngine()
        latest_vol = vol.iloc[[-1]]
        latest_regime = pd.Series([regimes.iloc[-1]], index=latest_vol.index)

        latest_weights = alloc_engine.compute_allocation(latest_vol, latest_regime)
        latest_weights = latest_weights.iloc[0].to_dict()

        # ---------------------------
        # 8️⃣ Stress Testing
        # ---------------------------
        stress_tester = StressTester()

        prices = fetch_price_data()
        asset_returns = compute_returns(prices)

        features = build_feature_set(prices)
        vol = compute_rolling_volatility(asset_returns)

        common_idx = asset_returns.index.intersection(features.index).intersection(vol.index)
        asset_returns = asset_returns.loc[common_idx]
        features = features.loc[common_idx]
        vol = vol.loc[common_idx]

        detector = RegimeDetector()
        regimes_full = detector.fit_predict(features)

        alloc_engine = AllocationEngine()
        weights = alloc_engine.compute_allocation(vol, regimes_full)

        stress_results = stress_tester.run_all_scenarios(
            asset_returns,
            weights,
            portfolio_returns
        )

        # ---------------------------
        # 9️⃣ Final Output
        # ---------------------------
        result = {
            "performance": performance_summary,
            "portfolio_returns": portfolio_returns,
            "regime_history": regimes,
            "volatility_series": vol,
            "latest_weights": latest_weights,
            "stress_test": stress_results,
            "risk_comparison": risk_comparison,  # 🔥 Added for dashboard
        }

        return self._to_python(result)


if __name__ == "__main__":
    service = PipelineService(window=120)
    output = service.run_pipeline()

    print("\n===== PIPELINE OUTPUT SUMMARY =====")
    print("Performance:", output["performance"])
    print("Latest Weights:", output["latest_weights"])
    print("Risk Comparison:", output["risk_comparison"])
    print("Stress Test:", output["stress_test"])
