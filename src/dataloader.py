import pandas as pd

from .config import Config

class DataLoader:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load(self) -> pd.DataFrame:
        """Load CSV"""

        df = pd.read_csv(self.cfg.DATA_PATH)
        print(f"[DataLoader] Raw file  : {df.shape[0]} rows, {df.shape[1]} cols")

        self._validate(df)
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        """Raise if required columns are missing."""
        
        assert self.cfg.TARGET in df.columns, f"Missing target: {self.cfg.TARGET}"

        for f in self.cfg.CONTINUOUS_FEATURES:
            assert f in df.columns, f"Missing feature: {f}"

        print(f"[DataLoader] Validated  : {df[self.cfg.TARGET].nunique()} classes")