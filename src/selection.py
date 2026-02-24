
import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

class ForwardSelection:
    """
    Greedy forward stepwise feature selection.
    Starts with no features, adds one at a time that gives the best score.
    """

    def __init__(self, estimator, scoring: str = "accuracy", cv: int = 5):
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.selected_features_: List[str] = []
        self.history_: List[dict] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, max_features: Optional[int] = None):
        """
        Run forward selection.

        Parameters
        ----------
        X : pd.DataFrame — feature matrix
        y : pd.Series    — target
        max_features : int or None — stop after this many (default: all)
        """

        all_features = list(X.columns)
        selected = []
        remaining = list(all_features)
        max_features = max_features or len(all_features)

        for step in range(min(max_features, len(all_features))):
            best_score = -np.inf
            best_feat = None

            for feat in remaining:
                candidate = selected + [feat]
                score = cross_val_score(
                    clone(self.estimator), X[candidate], y,
                    cv=self.cv, scoring=self.scoring,
                ).mean()

                if score > best_score:
                    best_score = score
                    best_feat = feat

            selected.append(best_feat)
            remaining.remove(best_feat)
            self.history_.append({
                "step": step + 1,
                "feature_added": best_feat,
                "features": list(selected),
                "score": best_score,
            })
            print(f"  Step {step+1}: +{best_feat:<45s} score={best_score:.4f}")

        self.selected_features_ = selected
        return self

    def get_best_features(self, n: Optional[int] = None) -> List[str]:
        """Return top-n features by the step with highest score."""

        if n is None:
            best_step = max(self.history_, key=lambda h: h["score"])
            return best_step["features"]
        
        return self.selected_features_[:n]


class BackwardSelection:
    """
    Greedy backward stepwise feature elimination.
    Starts with all features, removes one at a time whose removal hurts least.
    """

    def __init__(self, estimator, scoring: str = "accuracy", cv: int = 5):
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.selected_features_: List[str] = []
        self.history_: List[dict] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, min_features: int = 1):
        """
        Run backward elimination.

        Parameters
        ----------
        X : pd.DataFrame — feature matrix
        y : pd.Series    — target
        min_features : int — stop when this many features remain
        """

        remaining = list(X.columns)

        # Baseline with all features
        base_score = cross_val_score(
            clone(self.estimator), X[remaining], y,
            cv=self.cv, scoring=self.scoring,
        ).mean()
        
        self.history_.append({
            "step": 0,
            "feature_removed": None,
            "features": list(remaining),
            "score": base_score,
        })
        print(f"  Step 0 : all {len(remaining)} features           score={base_score:.4f}")

        step = 0
        while len(remaining) > min_features:
            step += 1
            best_score = -np.inf
            worst_feat = None

            for feat in remaining:
                candidate = [f for f in remaining if f != feat]
                score = cross_val_score(
                    clone(self.estimator), X[candidate], y,
                    cv=self.cv, scoring=self.scoring,
                ).mean()

                if score > best_score:
                    best_score = score
                    worst_feat = feat

            remaining.remove(worst_feat)
            self.history_.append({
                "step": step,
                "feature_removed": worst_feat,
                "features": list(remaining),
                "score": best_score,
            })
            print(f"  Step {step}: -{worst_feat:<45s} score={best_score:.4f}")

        self.selected_features_ = remaining
        return self

    def get_best_features(self) -> List[str]:
        """Return the feature set from the step with the highest score."""

        best_step = max(self.history_, key=lambda h: h["score"])
        return best_step["features"]

