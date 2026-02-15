import pandas as pd
import numpy as np


class AllocationEngine:
    def __init__(self):
        pass

    def risk_parity_weights(self, vol_row: pd.Series) -> pd.Series:
        # Prevent division explosion when volatility is near zero
        vol_row = vol_row.replace(0, 1e-6)

        inv_vol = 1 / vol_row
        weights = inv_vol / inv_vol.sum()
        return weights

    def regime_adjustment(self, weights: pd.Series, regime: int) -> pd.Series:
        """
        Adjust allocation based on detected regime.
        
        Regime Interpretation:
        0 -> Bull (increase equities)
        1 -> Bear (reduce equities)
        2 -> High Volatility / Stress (move to gold)
        """
        adjusted = weights.copy()

        if regime == 0:  # Bull Market
            adjusted["NIFTY"] *= 1.2
            adjusted["BANKNIFTY"] *= 1.2
            adjusted["GOLD"] *= 0.8

        elif regime == 1:  # Bear Market
            adjusted["NIFTY"] *= 0.7
            adjusted["BANKNIFTY"] *= 0.7
            adjusted["GOLD"] *= 1.3

        elif regime == 2:  # High Volatility / Crisis
            adjusted["NIFTY"] *= 0.5
            adjusted["BANKNIFTY"] *= 0.5
            adjusted["GOLD"] *= 1.8

        # Normalize weights to sum to 1
        adjusted = adjusted / adjusted.sum()

        return adjusted

    def compute_allocation(
        self,
        vol_df: pd.DataFrame,
        regimes: pd.Series
    ) -> pd.DataFrame:
        """
        Generate dynamic portfolio weights over time.
        """
        weights_list = []

        for date in vol_df.index:
            vol_row = vol_df.loc[date]
            regime = regimes.loc[date]

            base_weights = self.risk_parity_weights(vol_row)
            final_weights = self.regime_adjustment(base_weights, regime)

            weights_list.append(final_weights)

        weights_df = pd.DataFrame(weights_list, index=vol_df.index)
        return weights_df


if __name__ == "__main__":
    from backend.core.data_loader import fetch_price_data
    # from data_loader import fetch_price_data
    from backend.core.feature_engineering import (
        build_feature_set,
        compute_returns,
        compute_rolling_volatility
    )
    # from feature_engineering import (
    #     build_feature_set,
    #     compute_returns,
    #     compute_rolling_volatility
    # )
    from backend.core.regime_detection import RegimeDetector
    # from regime_detection import RegimeDetector

    prices = fetch_price_data()

    # Build features for regime detection
    features = build_feature_set(prices)

    # Compute returns & volatility separately for allocation
    returns = compute_returns(prices)
    vol = compute_rolling_volatility(returns)

    # Align indices
    common_index = vol.index.intersection(features.index)
    vol = vol.loc[common_index]
    features = features.loc[common_index]

    # Regime detection
    detector = RegimeDetector()
    regimes = detector.fit_predict(features)

    # Allocation
    engine = AllocationEngine()
    weights = engine.compute_allocation(vol, regimes)

    print(weights.head())
    print("\nSum of weights (should be 1):")
    print(weights.sum(axis=1).head())
