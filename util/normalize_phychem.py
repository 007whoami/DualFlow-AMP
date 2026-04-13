import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class AMPFeatureNormalizer:
    def __init__(self):
        self.scalers = {}
        self.feature_config = {
            "Sequence_Length":      {"method": "zscore", "scaler": StandardScaler()},
            "Net_Charge":           {"method": "zscore", "scaler": StandardScaler()},
            "Charge_Density":       {"method": "zscore", "scaler": StandardScaler()},
            "GRAVY":                {"method": "zscore", "scaler": StandardScaler()},
            "Hydrophobic_Moment":   {"method": "zscore", "scaler": StandardScaler()},
            "Aromatic_Ratio":       {"method": "zscore", "scaler": StandardScaler()},
            "Proline_Ratio":        {"method": "zscore", "scaler": StandardScaler()},
            "Glycine_Ratio":        {"method": "zscore", "scaler": StandardScaler()},
        }
        self.feature_order = list(self.feature_config.keys())
    
    def fit(self, df: pd.DataFrame):
        print("Fitting normalizer on training data...")
        for feat, config in self.feature_config.items():
            values = df[feat].values.reshape(-1, 1) #type:ignore
            config["scaler"].fit(values)
            self.scalers[feat] = config["scaler"]
            print(f"  {feat}: {config['method']}")
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        normalized_cols = []
        for feat in self.feature_order:
            values = df[feat].values.reshape(-1, 1) #type:ignore
            config = self.feature_config[feat]
            normalized_cols.append(config["scaler"].transform(values))
        return np.hstack(normalized_cols)
    
    def inverse_transform(self, normalized_array: np.ndarray) -> pd.DataFrame:
        result = {}
        col_idx = 0
        for feat in self.feature_order:
            config = self.feature_config[feat]
            values_norm = normalized_array[:, col_idx:col_idx+1]
            col_idx += 1
            result[feat] = config["scaler"].inverse_transform(values_norm).flatten()
        return pd.DataFrame(result)
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scalers": self.scalers, "config": self.feature_config, "order": self.feature_order}, path)
        print(f"Normalizer saved: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'AMPFeatureNormalizer':
        instance = cls()
        data = joblib.load(path)
        instance.scalers = data["scalers"]
        instance.feature_config = data["config"]
        instance.feature_order = data["order"]
        print(f"Normalizer loaded: {path}")
        return instance