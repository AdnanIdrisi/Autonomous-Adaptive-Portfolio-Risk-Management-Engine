import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class RegimeDetector:
    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes = n_regimes
        self.model = KMeans(n_clusters=n_regimes, random_state=random_state)
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, feature_df: pd.DataFrame):
        """
        Fit KMeans on feature set to learn market regimes.
        """
        scaled_features = self.scaler.fit_transform(feature_df)
        self.model.fit(scaled_features)
        self.fitted = True

    def predict(self, feature_df: pd.DataFrame) -> pd.Series:
        """
        Predict regime labels for each time step.
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        scaled_features = self.scaler.transform(feature_df)
        regimes = self.model.predict(scaled_features)

        return pd.Series(regimes, index=feature_df.index, name="Regime")

    def fit_predict(self, feature_df: pd.DataFrame) -> pd.Series:
        """
        Fit and predict regimes in one step.
        """
        self.fit(feature_df)
        return self.predict(feature_df)

if __name__ == "__main__":
    from backend.core.data_loader import fetch_price_data
    # from data_loader import fetch_price_data
    from backend.core.feature_engineering import build_feature_set
    # from feature_engineering import build_feature_set

    prices = fetch_price_data()
    features = build_feature_set(prices)

    detector = RegimeDetector(n_regimes=3)
    regimes = detector.fit_predict(features)

    print(regimes.value_counts())
    print(regimes.head())
