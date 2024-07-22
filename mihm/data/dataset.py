import pandas as pd
from typing import Sequence, Callable, Union, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

numeric = Union[int, float, complex, np.number]


class MIHMDataset:
    def __init__(
        self,
        df: pd.DataFrame,
        interactor: str,
        controlled_predictors: Sequence[str],
        interaction_predictors: Sequence[str],
        outcome: str,
        mean_std_dict: dict = {},
    ) -> None:
        self.df: pd.DataFrame = df
        self.columns: list = df.columns.tolist()
        self.interactor: str = interactor
        self.controlled_predictors: Sequence[str] = controlled_predictors
        self.interaction_predictors: Sequence[str] = interaction_predictors
        self.outcome: str = outcome
        self.mean_std_dict: dict = {}

    def __len__(self) -> int:
        return len(self.df)

    def preprocess(
        self,
        preprocess_list: Sequence[Tuple[Sequence[str], Callable]],
        inplace: bool = True,
    ):
        df_temp = self.df.copy()
        for colnames, preprocess in preprocess_list:
            df_temp = preprocess(df_temp, colnames)
        if inplace:
            self.df = df_temp
            return df_temp
        else:
            return df_temp

    def standardize(
        self,
        standardization_list: Sequence[Tuple[Sequence[str], Callable]],
        inplace: bool = True,
    ):
        df_temp = self.df.copy()
        for colnames, standardize in standardization_list:
            unprocessed_cols = []
            for col in colnames:
                if col not in self.mean_std_dict.keys():
                    unprocessed_cols.append(col)
            df_temp, mean_std_dict = standardize(df_temp, unprocessed_cols)
            # combine mean_std_dict
            self.mean_std_dict.update(mean_std_dict)
        if inplace:
            self.df = df_temp
            return df_temp
        else:
            return df_temp

    def reverse_standardize(
        self,
        colname: Union[None, Sequence, str] = None,
        value: Union[None, Sequence[Sequence[numeric]]] = None,
        inplace: bool = True,
    ) -> Union[pd.DataFrame, Sequence, numeric]:
        assert self.mean_std_dict is not None, "mean_std_dict is not available"
        if colname is not None and isinstance(colname, Sequence):
            assert all(
                [col in self.mean_std_dict for col in colname]
            ), "colname not in mean_std_dict"
        elif colname is not None and isinstance(colname, str):
            assert colname in self.mean_std_dict, "colname not in mean_std_dict"

        if value is None:
            df_temp = self.df.copy()
        else:
            df_temp = []

        if colname is None:
            assert value is None, "value must be None if colname is None"
            for col in self.mean_std_dict:
                mean, std = self.mean_std_dict[col]
                df_temp[col] = df_temp[col] * std + mean

        elif isinstance(colname, Sequence):
            if value is None:
                df_temp = df_temp[colname]
                for col in colname:
                    mean, std = self.mean_std_dict[col]
                    df_temp[col] = df_temp[col] * std + mean
            else:
                assert len(colname) == len(
                    value
                ), "colname and value must have same length"
                for col, val in zip(colname, value):
                    mean, std = self.mean_std_dict[col]
                    df_temp.append(val * std + mean)

        elif isinstance(colname, str):
            if value is None:
                df_temp = df_temp[colname]
                mean, std = self.mean_std_dict[colname]
                df_temp = df_temp * std + mean
            else:
                assert len(value) == 1, "value must have length 1"
                df_temp = [
                    v * self.mean_std_dict[colname][1] + self.mean_std_dict[colname][0]
                    for v in value[0]
                ]
        else:
            raise ValueError("colname must be None, str, or Sequence[str]")

        if value is None:
            if inplace:
                if colname is None:
                    self.df = df_temp
                else:
                    self.df[colname] = df_temp
            else:
                return df_temp
        else:
            return df_temp

    def dropna(self, inplace: bool = True):
        if inplace:
            self.df = self.df.dropna()
        else:
            return self.df.dropna()

    def __getitem__(self, idx: int):
        item_dict = {
            "interactor": self.df[self.interactor].iloc[idx].to_numpy(),
            "controlled": self.df[self.controlled_predictors].iloc[idx].to_numpy(),
            "interaction": self.df[self.interaction_predictors].iloc[idx].to_numpy(),
            "outcome": self.df[self.outcome].iloc[idx].to_numpy(),
        }
        return item_dict

    def __repr__(self):
        return f"MIHMDataset with {len(self)} samples"

    def get_subset(self, indices: Sequence[int]) -> "MIHMDataset":
        dataset_subset = MIHMDataset(
            self.df.iloc[indices],
            self.interactor,
            self.controlled_predictors,
            self.interaction_predictors,
            self.outcome,
            mean_std_dict=self.mean_std_dict,
        )
        # set extra attributes
        all_attrs_new = list(dataset_subset.__dict__.keys())
        for key, value in self.__dict__.items():
            if key not in all_attrs_new:
                setattr(dataset_subset, key, value)
        return dataset_subset

    def to_numpy(self, dtype=np.float32):
        item_dict = {
            "interactor": self.df[self.interactor].to_numpy().astype(dtype),
            "controlled_predictors": self.df[self.controlled_predictors]
            .to_numpy()
            .astype(dtype),
            "interaction_predictors": self.df[self.interaction_predictors]
            .to_numpy()
            .astype(dtype),
            "outcome": self.df[self.outcome].to_numpy().astype(dtype),
        }
        return item_dict

    def to_tensor(self, dtype=torch.float32, device="cpu"):
        item_dict = {
            "interactor": torch.tensor(
                self.df[self.interactor].to_numpy(), dtype=dtype
            ).to(device),
            "controlled_predictors": torch.tensor(
                self.df[self.controlled_predictors].to_numpy(), dtype=dtype
            ).to(device),
            "interaction_predictors": torch.tensor(
                self.df[self.interaction_predictors].to_numpy(), dtype=dtype
            ).to(device),
            "outcome": torch.tensor(self.df[self.outcome].to_numpy(), dtype=dtype).to(
                device
            ),
        }
        return item_dict

    def get_column_index(self, colname: str):
        return self.columns.index(colname)

    def to_torch_dataset(self, device="cpu"):
        return TorchMIHMDataset(
            self.df[self.interactor].to_numpy(),
            self.df[self.controlled_predictors].to_numpy(),
            self.df[self.interaction_predictors].to_numpy(),
            self.df[self.outcome].to_numpy(),
            device=device,
        )


class TorchMIHMDataset(Dataset):
    def __init__(
        self,
        interactor_var,
        controlled_vars,
        interaction_input_vars,
        label,
        device="cpu",
    ):
        self.interactor_var = interactor_var.astype(np.float32)
        self.controlled_vars = controlled_vars.astype(np.float32)
        self.interaction_input_vars = interaction_input_vars.astype(np.float32)
        self.label = label.astype(np.float32)
        self.device = device

    def __len__(self):
        return self.controlled_vars.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.to_list()

        sample = {
            "interactor": torch.tensor(self.interactor_var[index]).to(self.device),
            "controlled_predictors": torch.from_numpy(
                self.controlled_vars[index, :]
            ).to(self.device),
            "interaction_predictors": torch.from_numpy(
                self.interaction_input_vars[index, :]
            ).to(self.device),
            "outcome": torch.tensor(self.label[index]).to(self.device),
        }
        return sample
