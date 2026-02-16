import pandas as pd
import yfinance as yf


ASSETS = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "GOLD": "GOLDBEES.NS"
}


def fetch_price_data(start="2015-01-01", end="2025-12-31") -> pd.DataFrame:
    """
    Fetch closing prices for Indian market assets and return
    a clean aligned dataframe.
    """
    data = []

    for name, ticker in ASSETS.items():
        print(f"Downloading {name} ({ticker})...")

        df = yf.download(ticker, start=start, end=end, progress=False)

        if df.empty:
            raise ValueError(f"Failed to download data for {ticker}")

        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

        asset_series = df[price_col]
        asset_series.name = name  # <-- FIXED

        data.append(asset_series)

    price_df = pd.concat(data, axis=1)

    rename_map = {ticker: name for name, ticker in ASSETS.items()}
    price_df.rename(columns=rename_map, inplace=True)

    price_df = price_df.dropna()
    return price_df



if __name__ == "__main__":
    df = fetch_price_data()
    print(df.head())
    print("\nShape:", df.shape)
