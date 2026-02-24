from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    f1_score
)

import numpy as np
from typing import Tuple

class DiscriminantAnalysis:
    """LDA, QDA, threshold tuning, ROC curves."""

    @staticmethod
    def fit_lda(X_train, X_test, y_train, y_test):
        """Fit Linear Discriminant Analysis."""

        model = LinearDiscriminantAnalysis()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return model, y_pred, accuracy_score(y_test, y_pred)

    @staticmethod
    def fit_qda(X_train, X_test, y_train, y_test):
        """Fit Quadratic Discriminant Analysis."""

        model = QuadraticDiscriminantAnalysis()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return model, y_pred, accuracy_score(y_test, y_pred)

    @staticmethod
    def find_best_threshold(
        model, X_test, y_test, start=0.10, stop=0.91, step=0.05
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Sweep thresholds and return optimal F1."""

        probs = model.predict_proba(X_test)[:, 1]
        thresholds = np.arange(start, stop, step)

        f1s = np.array(
            [f1_score(y_test, (probs >= t).astype(int)) for t in thresholds]
        )
        best_idx = int(np.argmax(f1s))
        
        return thresholds[best_idx], f1s[best_idx], thresholds, f1s