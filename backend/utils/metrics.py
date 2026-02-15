import numpy as np
import pandas as pd


class PortfolioMetrics:
    def __init__(self, returns: pd.Series, risk_free_rate: float = 0.05):
        """
        returns: pd.Series of daily portfolio returns
        risk_free_rate: annual risk-free rate (default 5%)
        """
        self.returns = returns.dropna()
        self.rf = risk_free_rate
        self.trading_days = 252

    # -----------------------------
    # Core Metrics
    # -----------------------------
    def total_return(self):
        return (1 + self.returns).prod() - 1

    def annual_return(self):
        total_ret = self.total_return()
        n_years = len(self.returns) / self.trading_days
        return (1 + total_ret) ** (1 / n_years) - 1

    def annual_volatility(self):
        return self.returns.std() * np.sqrt(self.trading_days)

    # -----------------------------
    # Risk-Adjusted Metrics
    # -----------------------------
    def sharpe_ratio(self):
        excess_returns = self.returns - (self.rf / self.trading_days)
        return (excess_returns.mean() / self.returns.std()) * np.sqrt(self.trading_days)

    def sortino_ratio(self):
        downside = self.returns[self.returns < 0]
        downside_std = downside.std() * np.sqrt(self.trading_days)
        excess_returns = self.returns.mean() * self.trading_days - self.rf
        return excess_returns / downside_std if downside_std != 0 else np.nan

    # -----------------------------
    # Drawdown Metrics
    # -----------------------------
    def max_drawdown(self):
        cumulative = (1 + self.returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    def drawdown_series(self):
        cumulative = (1 + self.returns).cumprod()
        peak = cumulative.cummax()
        return (cumulative - peak) / peak

    # -----------------------------
    # Summary Report
    # -----------------------------
    def summary(self):
        return {
            "Total Return": self.total_return(),
            "Annual Return": self.annual_return(),
            "Annual Volatility": self.annual_volatility(),
            "Sharpe Ratio": self.sharpe_ratio(),
            "Sortino Ratio": self.sortino_ratio(),
            "Max Drawdown": self.max_drawdown(),
        }
