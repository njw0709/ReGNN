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
        interaction_predictors: Union[Sequence[str], Sequence[Sequence[str]]],
        outcome: str,
        survey_weights: Union[str, None] = None,
        mean_std_dict: dict = {},
    ) -> None:
        self.df: pd.DataFrame = df
        self.columns: list = df.columns.tolist()
        self.interactor: str = interactor
        self.controlled_predictors: Sequence[str] = controlled_predictors
        self.interaction_predictors: Union[Sequence[str], Sequence[Sequence[str]]] = interaction_predictors
        self.outcome: str = outcome
        self.survey_weights: Union[str, None] = survey_weights
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
            df_temp, new_colnames = preprocess(df_temp, colnames)
            if set(new_colnames) != set(colnames):
                for c in colnames:
                    lists_to_check = [self.controlled_predictors, self.interaction_predictors]
                    for current_list in lists_to_check:
                        if c in current_list:
                            current_list.remove(c)
                            current_list.extend([new_c for new_c in new_colnames if c in new_c])                          
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
        if isinstance(self.interaction_predictors[0], str):
            item_dict = {
                "interactor": self.df[self.interactor].iloc[idx].to_numpy(),
                "controlled": self.df[self.controlled_predictors].iloc[idx].to_numpy(),
                "interaction": self.df[self.interaction_predictors].iloc[idx].to_numpy(),
                "outcome": self.df[self.outcome].iloc[idx].to_numpy(),
                # "weights": self.df[self.survey_weights].iloc[idx].to_numpy(),
            }
        elif isinstance(self.interaction_predictors[0], list):
            item_dict = {
                "interactor": self.df[self.interactor].iloc[idx].to_numpy(),
                "controlled": self.df[self.controlled_predictors].iloc[idx].to_numpy(),
                "interaction": [self.df[i_p].iloc[idx].to_numpy() for i_p in self.interaction_predictors],
                "outcome": self.df[self.outcome].iloc[idx].to_numpy(),
                # "weights": self.df[self.survey_weights].iloc[idx].to_numpy(),
            }
        else:
            raise TypeError("moderator must either be string or list of strings")

        if self.survey_weights is not None:
            item_dict["weights"] = self.df[self.survey_weights].iloc[idx].to_numpy()

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
            self.survey_weights,
            mean_std_dict=self.mean_std_dict,
        )
        # set extra attributes
        all_attrs_new = list(dataset_subset.__dict__.keys())
        for key, value in self.__dict__.items():
            if key not in all_attrs_new:
                setattr(dataset_subset, key, value)
        return dataset_subset

    def to_numpy(self, dtype=np.float32):
        if isinstance(self.interaction_predictors[0], str):
            item_dict = {
                "interactor": self.df[self.interactor].to_numpy().astype(dtype),
                "controlled_predictors": self.df[self.controlled_predictors]
                .to_numpy()
                .astype(dtype),
                "interaction_predictors": self.df[self.interaction_predictors]
                .to_numpy()
                .astype(dtype),
                "outcome": self.df[self.outcome].to_numpy().astype(dtype),
                # "weights": self.df[self.survey_weights].to_numpy().astype(dtype)
            }
        elif isinstance(self.interaction_predictors[0], list):
            item_dict = {
                "interactor": self.df[self.interactor].to_numpy().astype(dtype),
                "controlled_predictors": self.df[self.controlled_predictors]
                .to_numpy()
                .astype(dtype),
                "interaction_predictors": [self.df[i_p]
                .to_numpy() 
                .astype(dtype) for i_p in self.interaction_predictors],
                "outcome": self.df[self.outcome].to_numpy().astype(dtype),
                # "weights": self.df[self.survey_weights].to_numpy().astype(dtype)
            }
        else:
            raise TypeError()
        if self.survey_weights is not None:
            item_dict["weights"] = self.df[self.survey_weights].to_numpy().astype(dtype)
        return item_dict

    def to_tensor(self, dtype=torch.float32, device="cpu"):
        if isinstance(self.interaction_predictors[0], str):
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
                # "weights": torch.tensor(self.df[self.survey_weights].to_numpy(), dtype=dtype).to(device)
            }
        elif isinstance(self.interaction_predictors[0], list):
            item_dict = {
                "interactor": torch.tensor(
                    self.df[self.interactor].to_numpy(), dtype=dtype
                ).to(device),
                "controlled_predictors": torch.tensor(
                    self.df[self.controlled_predictors].to_numpy(), dtype=dtype
                ).to(device),
                "interaction_predictors": [torch.tensor(
                    self.df[i_p].to_numpy(), dtype=dtype
                ).to(device) for i_p in self.interaction_predictors],
                "outcome": torch.tensor(self.df[self.outcome].to_numpy(), dtype=dtype).to(
                    device
                ),
                # "weights": torch.tensor(self.df[self.survey_weights].to_numpy(), dtype=dtype).to(device)
            }
        else:
            raise TypeError()
        if self.survey_weights is not None:
            item_dict["weights"] = torch.tensor(self.df[self.survey_weights].to_numpy(), dtype=dtype).to(device)
        return item_dict

    def get_column_index(self, colname: str):
        return self.columns.index(colname)

    def to_torch_dataset(self, device="cpu"):
        if self.survey_weights is not None:
            weights = self.df[self.survey_weights].to_numpy()
        else:
            weights = None
        if isinstance(self.interaction_predictors[0], str):
            return TorchMIHMDataset(
                self.df[self.interactor].to_numpy(),
                self.df[self.controlled_predictors].to_numpy(),
                self.df[self.interaction_predictors].to_numpy(),
                self.df[self.outcome].to_numpy(),
                weights,
                device=device,
            )
        elif isinstance(self.interaction_predictors[0], list):
            return TorchMIHMDataset(
                self.df[self.interactor].to_numpy(),
                self.df[self.controlled_predictors].to_numpy(),
                [self.df[i_p].to_numpy() for i_p in self.interaction_predictors],
                self.df[self.outcome].to_numpy(),
                weights,
                device=device,
            )
        else:
            raise TypeError()

class TorchMIHMDataset(Dataset):
    def __init__(
        self,
        interactor_var,
        controlled_vars,
        interaction_input_vars,
        label,
        weights: Union[np.ndarray, None] = None,
        device="cpu",
    ):
        self.interactor_var = interactor_var.astype(np.float32)
        self.controlled_vars = controlled_vars.astype(np.float32)
        if isinstance(interaction_input_vars, list):
            self.interaction_input_vars = [i_p.astype(np.float32) for i_p in interaction_input_vars]
        else:
            self.interaction_input_vars = interaction_input_vars.astype(np.float32)
        self.label = label.astype(np.float32)
        if weights is not None:
            self.weights = weights.astype(np.float32)
        else:
            self.weights = None
        self.device = device

    def __len__(self):
        return self.controlled_vars.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.to_list()
        if isinstance(self.interaction_input_vars, list):
            sample = {
                "interactor": torch.tensor(self.interactor_var[index]).to(self.device),
                "controlled_predictors": torch.from_numpy(
                    self.controlled_vars[index, :]
                ).to(self.device),
                "interaction_predictors": [torch.from_numpy(
                    i_p[index, :]
                ).to(self.device) for i_p in self.interaction_input_vars],
                "outcome": torch.tensor(self.label[index]).to(self.device),
                # "weights": torch.tensor(self.weights[index]).to(self.device)
            }
        else:
            sample = {
                "interactor": torch.tensor(self.interactor_var[index]).to(self.device),
                "controlled_predictors": torch.from_numpy(
                    self.controlled_vars[index, :]
                ).to(self.device),
                "interaction_predictors": torch.from_numpy(
                    self.interaction_input_vars[index, :]
                ).to(self.device),
                "outcome": torch.tensor(self.label[index]).to(self.device),
                # "weights": torch.tensor(self.weights[index]).to(self.device)
            }
        if self.weights is not None:
            sample["weights"] = torch.tensor(self.weights[index]).to(self.device)
        return sample
