from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    f1_score
)

import pandas as pd
from typing import Dict, Tuple

class NaiveBayesAnalysis:

    @staticmethod
    def fit(X_train, X_test, y_train, y_test):
        """Fit Gaussian Naive Bayes."""
        
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, y_pred, accuracy_score(y_test, y_pred)

    @staticmethod
    def compare_models(models: Dict[str, Tuple], y_test) -> pd.DataFrame:
        """Build comparison DataFrame from {name: (model, y_pred)} dict."""

        rows = []
        for name, (_, preds) in models.items():
            rows.append(
                {
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, preds),
                    "F1_Weighted": f1_score(y_test, preds, average="weighted"),
                    "F1_Macro": f1_score(y_test, preds, average="macro"),
                }
            )

        return pd.DataFrame(rows)