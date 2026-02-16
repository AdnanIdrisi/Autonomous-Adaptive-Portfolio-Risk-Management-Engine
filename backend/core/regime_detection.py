import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM


class RegimeDetector:
    """
    Hidden Markov Model based regime detection.
    Supports:
    - fit()
    - predict()
    - fit_predict()
    """

    def __init__(self, n_states: int = 2):
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42
        )
        self.fitted = False
        self.bull_state = None

    # --------------------------------------------------
    # Fit model on training features
    # --------------------------------------------------
    def fit(self, features: pd.DataFrame):
        X = features.values
        self.model.fit(X)

        hidden_states = self.model.predict(X)

        # Determine which state is bull (higher mean return)
        state_means = []
        for i in range(self.n_states):
            if np.sum(hidden_states == i) > 0:
                state_means.append(X[hidden_states == i][:, 0].mean())
            else:
                state_means.append(-np.inf)

        self.bull_state = int(np.argmax(state_means))
        self.fitted = True


    # --------------------------------------------------
    # Predict regime for new data
    # --------------------------------------------------
    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction.")

        X = features.values
        hidden_states = self.model.predict(X)

        regimes = pd.Series(hidden_states, index=features.index)
        regimes = regimes.apply(lambda x: 1 if x == self.bull_state else 0)

        return regimes

    # --------------------------------------------------
    # Fit and predict in one step (for pipeline use)
    # --------------------------------------------------
    def fit_predict(self, features: pd.DataFrame) -> pd.Series:
        self.fit(features)
        return self.predict(features)
