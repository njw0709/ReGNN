import pandas as pd
from typing import Sequence, Union, Tuple, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from .base import BaseDataset, ReGNNDatasetConfig
from .preprocessor_mixin import PreprocessorMixin


class ReGNNDataset(BaseDataset, PreprocessorMixin):
    """Main dataset class for ReGNN models"""

    def __init__(
        self,
        df: pd.DataFrame,
        config: ReGNNDatasetConfig,
    ) -> None:
        df, df_orig = self._initial_processing(df, config.df_dtypes, config.rename_dict)
        self.df_orig = df_orig
        super().__init__(df, config)
        # self.mean_std_dict = mean_std_dict # move to processing_step

    def _initial_processing(
        self, df: pd.DataFrame, df_dtypes: Dict[str, str], rename_dict: Dict[str, str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_orig = df.copy()
        df = df.dropna().reset_index(drop=True).rename(columns=rename_dict)

        # get binary and multi category columns
        for dtype, c in df_dtypes.items():
            if dtype == "category" or dtype == "binary":
                df[c] = df[c].astype("category")

        return df, df_orig

    def __getitem__(self, idx: int):
        if isinstance(self.config.moderators[0], str):
            item_dict = {
                "focal_predictor": np.array(
                    [self.df[self.config.focal_predictor].iloc[idx]]
                ),
                "controlled": self.df[self.config.controlled_predictors]
                .iloc[idx]
                .values,
                "moderators": self.df[self.config.moderators].iloc[idx].values,
                "outcome": np.array([self.df[self.config.outcome].iloc[idx]]),
            }
        elif isinstance(self.config.moderators[0], list):
            item_dict = {
                "focal_predictor": np.array(
                    [self.df[self.config.focal_predictor].iloc[idx]]
                ),
                "controlled": self.df[self.config.controlled_predictors]
                .iloc[idx]
                .values,
                "moderators": [
                    np.array([self.df[i_p].iloc[idx]]) for i_p in self.config.moderators
                ],
                "outcome": np.array([self.df[self.config.outcome].iloc[idx]]),
            }
        else:
            raise TypeError("moderator must either be string or list of strings")

        if self.config.survey_weights is not None:
            item_dict["weights"] = np.array(
                [self.df[self.config.survey_weights].iloc[idx]]
            )

        return item_dict

    def __repr__(self):
        return f"ReGNNDataset with {len(self)} samples"

    def get_subset(self, indices: Sequence[int]) -> "ReGNNDataset":
        dataset_subset = ReGNNDataset(
            self.df_orig.iloc[indices],
            config=self.config,
        )
        # preprocess
        dataset_subset.preprocess()

        # set extra attributes
        all_attrs_new = list(dataset_subset.__dict__.keys())
        for key, value in self.__dict__.items():
            if key not in all_attrs_new:
                setattr(dataset_subset, key, value)
        return dataset_subset

    def to_numpy(self, dtype=np.float32):
        if isinstance(self.config.moderators[0], str):
            item_dict = {
                "focal_predictor": self.df[self.config.focal_predictor]
                .to_numpy()
                .astype(dtype),
                "controlled_predictors": self.df[self.config.controlled_predictors]
                .to_numpy()
                .astype(dtype),
                "moderators": self.df[self.config.moderators].to_numpy().astype(dtype),
                "outcome": self.df[self.config.outcome].to_numpy().astype(dtype),
            }
        elif isinstance(self.config.moderators[0], list):
            item_dict = {
                "focal_predictor": self.df[self.config.focal_predictor]
                .to_numpy()
                .astype(dtype),
                "controlled_predictors": self.df[self.config.controlled_predictors]
                .to_numpy()
                .astype(dtype),
                "moderators": [
                    self.df[i_p].to_numpy().astype(dtype)
                    for i_p in self.config.moderators
                ],
                "outcome": self.df[self.config.outcome].to_numpy().astype(dtype),
            }
        else:
            raise TypeError()
        if self.config.survey_weights is not None:
            item_dict["weights"] = (
                self.df[self.config.survey_weights].to_numpy().astype(dtype)
            )
        return item_dict

    def to_tensor(self, dtype=torch.float32, device="cpu"):
        if isinstance(self.config.moderators[0], str):
            item_dict = {
                "focal_predictor": torch.tensor(
                    self.df[self.config.focal_predictor].to_numpy(), dtype=dtype
                ).to(device),
                "controlled_predictors": torch.tensor(
                    self.df[self.config.controlled_predictors].to_numpy(), dtype=dtype
                ).to(device),
                "moderators": torch.tensor(
                    self.df[self.config.moderators].to_numpy(), dtype=dtype
                ).to(device),
                "outcome": torch.tensor(
                    self.df[self.config.outcome].to_numpy(), dtype=dtype
                ).to(device),
            }
        elif isinstance(self.config.moderators[0], list):
            item_dict = {
                "focal_predictor": torch.tensor(
                    self.df[self.config.focal_predictor].to_numpy(), dtype=dtype
                ).to(device),
                "controlled_predictors": torch.tensor(
                    self.df[self.config.controlled_predictors].to_numpy(), dtype=dtype
                ).to(device),
                "moderators": [
                    torch.tensor(self.df[i_p].to_numpy(), dtype=dtype).to(device)
                    for i_p in self.config.moderators
                ],
                "outcome": torch.tensor(
                    self.df[self.config.outcome].to_numpy(), dtype=dtype
                ).to(device),
            }
        else:
            raise TypeError()
        if self.config.survey_weights is not None:
            item_dict["weights"] = torch.tensor(
                self.df[self.config.survey_weights].to_numpy(), dtype=dtype
            ).to(device)
        return item_dict

    def to_torch_dataset(self, device="cpu"):
        if self.config.survey_weights is not None:
            weights = self.df[self.config.survey_weights].to_numpy()
        else:
            weights = None
        if isinstance(self.config.moderators[0], str):
            return TorchReGNNDataset(
                self.df[self.config.focal_predictor].to_numpy(),
                self.df[self.config.controlled_predictors].to_numpy(),
                self.df[self.config.moderators].to_numpy(),
                self.df[self.config.outcome].to_numpy(),
                weights,
                device=device,
            )
        elif isinstance(self.config.moderators[0], list):
            return TorchReGNNDataset(
                self.df[self.config.focal_predictor].to_numpy(),
                self.df[self.config.controlled_predictors].to_numpy(),
                [self.df[i_p].to_numpy() for i_p in self.config.moderators],
                self.df[self.config.outcome].to_numpy(),
                weights,
                device=device,
            )
        else:
            raise TypeError()


class TorchReGNNDataset(Dataset):
    def __init__(
        self,
        focal_predictor_var,
        controlled_vars,
        moderator_vars,
        label,
        weights: Optional[np.ndarray] = None,
        device="cpu",
    ):
        self.focal_predictor_var = focal_predictor_var.astype(np.float32)
        self.controlled_vars = controlled_vars.astype(np.float32)
        if isinstance(moderator_vars, list):
            self.moderator_vars = [i_p.astype(np.float32) for i_p in moderator_vars]
        else:
            self.moderator_vars = moderator_vars.astype(np.float32)
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
        if isinstance(self.moderator_vars, list):
            sample = {
                "focal_predictor": torch.tensor(self.focal_predictor_var[index]).to(
                    self.device
                ),
                "controlled_predictors": torch.from_numpy(
                    self.controlled_vars[index, :]
                ).to(self.device),
                "moderators": [
                    torch.from_numpy(i_p[index, :]).to(self.device)
                    for i_p in self.moderator_vars
                ],
                "outcome": torch.tensor(self.label[index]).to(self.device),
            }
        else:
            sample = {
                "focal_predictor": torch.tensor(self.focal_predictor_var[index]).to(
                    self.device
                ),
                "controlled_predictors": torch.from_numpy(
                    self.controlled_vars[index, :]
                ).to(self.device),
                "moderators": torch.from_numpy(self.moderator_vars[index, :]).to(
                    self.device
                ),
                "outcome": torch.tensor(self.label[index]).to(self.device),
            }
        if self.weights is not None:
            sample["weights"] = torch.tensor(self.weights[index]).to(self.device)
        return sample
