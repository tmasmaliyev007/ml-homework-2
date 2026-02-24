
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy import stats

from ..config import Config

import pandas as pd
import numpy as np

from typing import Tuple, List

class LogisticRegressionAnalysis:
    """Wraps binary & multi-class LR plus coefficient inference."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def fit_binary(
        self, X_train, X_test, y_train, y_test
    ) -> Tuple[LogisticRegression, np.ndarray, float]:
        """Fit binary logistic regression and return model, predictions, accuracy."""

        model = LogisticRegression(
            max_iter=self.cfg.LR_MAX_ITER, random_state=self.cfg.RANDOM_STATE
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        return model, y_pred, acc

    def coefficient_statistics(
        self, model: LogisticRegression, X_train: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute standard errors, Z-statistics, p-values via Fisher Information."""

        coefs = np.concatenate([model.intercept_, model.coef_[0]])
        probs = model.predict_proba(X_train)[:, 1]

        # Avoid creating n×n diagonal matrix
        weights = probs * (1 - probs)  # shape (n,)
        X_des = np.column_stack([np.ones(X_train.shape[0]), X_train.values])

        X_weighted = X_des * weights[:, np.newaxis]  # broadcast: (n, p) * (n, 1)
        fisher = X_weighted.T @ X_des  # (p, p)

        se = np.sqrt(np.diag(np.linalg.inv(fisher)))
        z = coefs / se
        p = 2 * (1 - stats.norm.cdf(np.abs(z)))

        return pd.DataFrame(
            {
                "Feature": ["Intercept"] + list(X_train.columns),
                "Coefficient": coefs,
                "Std_Error": se,
                "Z_Statistic": z,
                "P_Value": p,
                "Significance": [
                    "***" if pv < 0.001
                    else "**" if pv < 0.01
                    else "*" if pv < 0.05
                    else ""
                    for pv in p
                ],
            }
        )

    @staticmethod
    def confounding_analysis(
        X: pd.DataFrame, features: List[str]
    ) -> Tuple[List[Tuple], pd.DataFrame]:
        """Correlation pairs + Variance Inflation Factors."""
        corr = X[features].corr()
        pairs = []

        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                c = corr.iloc[i, j]
                if abs(c) > 0.5:
                    pairs.append((features[i], features[j], round(c, 3)))

        vals = X[features].values
        X_std = (vals - vals.mean(0)) / vals.std(0)
        vif = np.diag(np.linalg.inv(np.corrcoef(X_std.T)))
        vif_df = pd.DataFrame(
            {"Feature": features, "VIF": vif}
        ).sort_values("VIF", ascending=False)

        return pairs, vif_df

    def fit_multiclass(
        self, X_train, X_test, y_train, y_test
    ) -> Tuple[LogisticRegression, np.ndarray, float]:
        """Fit multinomial logistic regression."""

        model = LogisticRegression(
            max_iter=self.cfg.LR_MAX_ITER, random_state=self.cfg.RANDOM_STATE
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        return model, y_pred, acc