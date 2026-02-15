import pandas as pd
import numpy as np


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change()

    # Remove inf values from bad data
    returns = returns.replace([np.inf, -np.inf], np.nan)

    # 🔴 Remove extreme impossible returns (data glitches)
    returns = returns[(returns.abs() < 0.5).all(axis=1)]

    returns = returns.dropna()
    return returns


def compute_rolling_volatility(returns_df: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    """Compute rolling annualized volatility."""
    vol = returns_df.rolling(window).std() * np.sqrt(252)
    return vol.dropna()


def compute_moving_averages(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 50-day and 120-day moving averages."""
    ma_50 = price_df.rolling(50).mean()
    ma_120 = price_df.rolling(120).mean()

    ma_df = pd.concat(
        [ma_50.add_suffix("_MA50"), ma_120.add_suffix("_MA120")],
        axis=1
    )

    return ma_df.dropna()


def compute_drawdown(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling drawdown for each asset."""
    cumulative_max = price_df.cummax()
    drawdown = (price_df - cumulative_max) / cumulative_max
    return drawdown


def compute_correlation(returns_df: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    """Compute rolling correlation matrix (flattened)."""
    corr_list = []

    for i in range(window, len(returns_df)):
        corr = returns_df.iloc[i-window:i].corr()
        corr_flat = corr.values[np.triu_indices_from(corr.values, k=1)]
        corr_list.append(corr_flat)

    corr_df = pd.DataFrame(corr_list, index=returns_df.index[window:])
    return corr_df


def build_feature_set(price_df: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    """
    Master function to create all features required for regime detection.
    """
    returns = compute_returns(price_df)
    volatility = compute_rolling_volatility(returns, window)
    moving_avg = compute_moving_averages(price_df)
    drawdown = compute_drawdown(price_df)

    # Align all features
    features = pd.concat(
        [returns, volatility.add_suffix("_VOL"), moving_avg, drawdown.add_suffix("_DD")],
        axis=1
    ).dropna()

    return features


if __name__ == "__main__":
    # from backend.core.data_loader import fetch_price_data
    from data_loader import fetch_price_data

    prices = fetch_price_data()
    features = build_feature_set(prices)

    print(features.head())
    print("\nFeature Shape:", features.shape)
