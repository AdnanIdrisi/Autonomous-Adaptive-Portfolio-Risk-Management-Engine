import pandas as pd
from backend.core.backtester import Backtester
from backend.utils.metrics import PortfolioMetrics


class RiskComparison:
    """
    Compare portfolio performance:
    1. Without Risk Management
    2. With Hybrid Risk Engine
    """

    def __init__(self, window: int = 120):
        self.window = window

    # --------------------------------------------------
    # Run backtest WITHOUT risk controls
    # --------------------------------------------------
    def run_without_risk(self):
        backtester = Backtester(window=self.window)

        # Temporarily disable risk manager
        original_apply = backtester.run_backtest

        def run_no_risk():
            # Monkey patch: bypass risk controls
            from backend.core.risk_manager import RiskManager

            original_rm = RiskManager.apply_risk_controls

            def identity(self, weights, returns):
                return weights  # Do nothing

            RiskManager.apply_risk_controls = identity
            result = original_apply()
            RiskManager.apply_risk_controls = original_rm  # Restore
            return result

        return run_no_risk()

    # --------------------------------------------------
    # Run backtest WITH risk controls
    # --------------------------------------------------
    def run_with_risk(self):
        backtester = Backtester(window=self.window)
        return backtester.run_backtest()

    # --------------------------------------------------
    # Compare both strategies
    # --------------------------------------------------
    def compare(self):
        print("\n===== Running Risk Engine Comparison =====")

        returns_no_risk = self.run_without_risk()
        returns_with_risk = self.run_with_risk()

        metrics_no_risk = PortfolioMetrics(returns_no_risk).summary()
        metrics_with_risk = PortfolioMetrics(returns_with_risk).summary()

        comparison = pd.DataFrame({
            "Without Risk Engine": metrics_no_risk,
            "With Risk Engine": metrics_with_risk
        })

        return comparison

if __name__ == "__main__":
    comparator = RiskComparison(window=120)
    comparison_result = comparator.compare()
    print(comparison_result)