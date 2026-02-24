from typing import Tuple
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import Config

class DataPreprocessor:
    """Feature scaling and train/test splitting."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.scaler_bin = StandardScaler()
        self.scaler_reg = StandardScaler()

    def prepare_multiclass(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Scaled X_train, X_test, y_train, y_test for all classes."""

        X = df[self.cfg.CONTINUOUS_FEATURES].copy()
        y = df[self.cfg.TARGET].copy()

        X_tr, X_t, y_tr, y_t = train_test_split(
            X, y, test_size=self.cfg.TEST_SIZE,
            random_state=self.cfg.RANDOM_STATE, stratify=y,
        )

        X_trs = pd.DataFrame(
            self.scaler.fit_transform(X_tr), columns=self.cfg.CONTINUOUS_FEATURES
        )
        X_ts = pd.DataFrame(
            self.scaler.transform(X_t), columns=self.cfg.CONTINUOUS_FEATURES
        )

        return X_trs, X_ts, y_tr, y_t


    def prepare_binary(
        self, df: pd.DataFrame, class_a: int = 1, class_b: int = 2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Binary subset (class_a → 1, class_b → 0)."""

        mask = df[self.cfg.TARGET].isin([class_a, class_b])
        X = df.loc[mask, self.cfg.CONTINUOUS_FEATURES].copy().reset_index(drop=True)
        y = (df.loc[mask, self.cfg.TARGET].reset_index(drop=True) == class_a).astype(int)

        X_tr, X_t, y_tr, y_t = train_test_split(
            X, y, test_size=self.cfg.TEST_SIZE,
            random_state=self.cfg.RANDOM_STATE, stratify=y,
        )
        
        X_trs = pd.DataFrame(
            self.scaler.fit_transform(X_tr), columns=self.cfg.CONTINUOUS_FEATURES
        )
        X_ts = pd.DataFrame(
            self.scaler.transform(X_t), columns=self.cfg.CONTINUOUS_FEATURES
        )

        return X_trs, X_ts, y_tr, y_t

    def prepare_regression(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Regression data (predict count-like target)."""
        feats = [
            f for f in self.cfg.CONTINUOUS_FEATURES
            if f != self.cfg.REGRESSION_TARGET
        ]

        X = df[feats].copy()
        y = df[self.cfg.REGRESSION_TARGET].copy()
        X_s = pd.DataFrame(self.scaler_reg.fit_transform(X), columns=feats)

        return train_test_split(
            X_s, y, test_size=self.cfg.TEST_SIZE,
            random_state=self.cfg.RANDOM_STATE,
        )