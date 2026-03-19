import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List, Tuple

class DataLoader:
    def __init__(
        self,
        file_path: str = "data_fine0.xlsx",
        list_sheet: str = "data",
        map_sheet: str = "scope",
        configs: List[Dict] = None,
        remove_duplicates: bool = True,
    ):
        self.file_path = file_path
        self.list_sheet = list_sheet
        self.map_sheet = map_sheet
        self.configs = configs or self._default_configs()
        self.remove_duplicates = remove_duplicates
        self.encoders = {}
        self.scalers_y = {}
        self.load_data()
        self.preprocess()

    def _default_configs(self) -> List[Dict]:
        return [
            {"name": "stage1", "input_cols": range(1, 16), "output_cols": [16]},
            {"name": "stage2", "input_cols": range(17, 23), "output_cols": [23, 24]},
            {"name": "stage3", "input_cols": range(25, 32), "output_cols": [32]},
            {"name": "stage4", "input_cols": [33], "output_cols": [34]},
        ]

    def load_data(self) -> None:
        try:
            self.list_df = pd.read_excel(self.file_path, sheet_name=self.list_sheet).dropna(how="all")
            self.map_df = pd.read_excel(self.file_path, sheet_name=self.map_sheet).dropna(how="all")
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")

    def preprocess(self) -> None:
        all_input_cols = []
        for cfg in self.configs:
            all_input_cols.extend(list(cfg["input_cols"]))

        list_columns = self.list_df.columns.tolist()
        map_columns = self.map_df.columns.tolist()

        self.categorical_cols = [c for c in all_input_cols if list_columns[c] in map_columns]
        self.numerical_cols = [c for c in all_input_cols if c not in self.categorical_cols]

        for col_idx in self.categorical_cols:
            le = LabelEncoder()
            self.list_df.iloc[:, col_idx] = le.fit_transform(self.list_df.iloc[:, col_idx].astype(str))
            self.encoders[col_idx] = le

        self.list_df.iloc[:, self.numerical_cols] = self.list_df.iloc[:, self.numerical_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    def get_features(self):
        if self.remove_duplicates:
            feature_cols = self.categorical_cols + self.numerical_cols
            feature_strings = self.list_df.iloc[:, feature_cols].astype(str).agg('_'.join, axis=1)
            unique_indices = feature_strings.drop_duplicates().index
            df_filtered = self.list_df.loc[unique_indices].copy()
        else:
            df_filtered = self.list_df.copy()

        scaler_x = StandardScaler()
        if self.numerical_cols:
            df_filtered.iloc[:, self.numerical_cols] = scaler_x.fit_transform(df_filtered.iloc[:, self.numerical_cols])

        X_dict = {}
        for config in self.configs:
            cols = list(config["input_cols"])
            cat_c = [c for c in cols if c in self.categorical_cols]
            num_c = [c for c in cols if c in self.numerical_cols]

            x_cat = df_filtered.iloc[:, cat_c].values.astype(np.int64) if cat_c else np.zeros((len(df_filtered), 0), dtype=np.int64)
            x_num = df_filtered.iloc[:, num_c].values.astype(np.float32) if num_c else np.zeros((len(df_filtered), 0), dtype=np.float32)

            X_dict[config["name"]] = {
                "cat": x_cat, "num": x_num,
                "cat_dims": [len(self.encoders[c].classes_) for c in cat_c],
                "num_dim": len(num_c)
            }

        Y_targets = {}
        for config in self.configs:
            output_cols = config.get("output_col", config.get("output_cols"))
            if isinstance(output_cols, int): output_cols = [output_cols]
            y_vals = df_filtered.iloc[:, output_cols].values.astype(np.float32)

            scaler_y = StandardScaler()
            Y_targets[config["name"]] = scaler_y.fit_transform(y_vals)
            self.scalers_y[config["name"]] = scaler_y

        return X_dict, Y_targets

class MultiTaskDataset(Dataset):
    def __init__(self, X_dict, Y_targets):
        self.length = len(list(Y_targets.values())[0])
        self.X = {}
        for stage, feats in X_dict.items():
            self.X[stage] = {
                "cat": torch.tensor(feats["cat"], dtype=torch.long),
                "num": torch.tensor(feats["num"], dtype=torch.float32)
            }
        self.Y = {stage: torch.tensor(Y_targets[stage], dtype=torch.float32) for stage in Y_targets}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_item = {stage: {"cat": feats["cat"][idx], "num": feats["num"][idx]} for stage, feats in self.X.items()}
        y_item = {stage: self.Y[stage][idx] for stage in self.Y}
        return x_item, y_item
