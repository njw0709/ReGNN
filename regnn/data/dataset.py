import pandas as pd
from typing import Sequence, Union, Tuple, Dict, Optional, Literal
import numpy as np
import torch
from torch.utils.data import Dataset
from .base import BaseDataset, ReGNNDatasetConfig
from .preprocessor_mixin import PreprocessorMixin


class ReGNNDataset(BaseDataset, PreprocessorMixin, Dataset):
    """Main dataset class for ReGNN models"""

    def __init__(
        self,
        df: pd.DataFrame,
        config: ReGNNDatasetConfig,
        output_mode: Literal["numpy", "tensor"] = "numpy",
        device: str = "cpu",
        dtype: Union[np.dtype, torch.dtype] = np.float32,
    ) -> None:
        df, df_orig = self._initial_processing(df, config.df_dtypes, config.rename_dict)
        self.df_orig = df_orig
        super().__init__(df, config)
        self.output_mode = output_mode
        self.device = device
        self.dtype = dtype

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

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        if self.output_mode == "tensor":
            if isinstance(self.config.moderators[0], list):
                return {
                    "focal_predictor": torch.tensor(
                        self.df[self.config.focal_predictor].iloc[idx].values,
                        dtype=self.dtype,
                    ).to(self.device),
                    "controlled_predictors": torch.tensor(
                        self.df[self.config.controlled_predictors].iloc[idx].values,
                        dtype=self.dtype,
                    ).to(self.device),
                    "moderators": [
                        torch.tensor(
                            self.df[i_p].iloc[idx].values, dtype=self.dtype
                        ).to(self.device)
                        for i_p in self.config.moderators
                    ],
                    "outcome": torch.tensor(
                        self.df[self.config.outcome].iloc[idx].values, dtype=self.dtype
                    ).to(self.device),
                    **(
                        {
                            "weights": torch.tensor(
                                self.df[self.config.survey_weights].iloc[idx].values,
                                dtype=self.dtype,
                            ).to(self.device)
                        }
                        if self.config.survey_weights is not None
                        else {}
                    ),
                }
            else:
                return {
                    "focal_predictor": torch.tensor(
                        self.df[self.config.focal_predictor].iloc[idx].values,
                        dtype=self.dtype,
                    ).to(self.device),
                    "controlled_predictors": torch.tensor(
                        self.df[self.config.controlled_predictors].iloc[idx].values,
                        dtype=self.dtype,
                    ).to(self.device),
                    "moderators": torch.tensor(
                        self.df[self.config.moderators].iloc[idx].values,
                        dtype=self.dtype,
                    ).to(self.device),
                    "outcome": torch.tensor(
                        self.df[self.config.outcome].iloc[idx].values, dtype=self.dtype
                    ).to(self.device),
                    **(
                        {
                            "weights": torch.tensor(
                                self.df[self.config.survey_weights].iloc[idx].values,
                                dtype=self.dtype,
                            ).to(self.device)
                        }
                        if self.config.survey_weights is not None
                        else {}
                    ),
                }
        else:  # numpy mode
            if isinstance(self.config.moderators[0], list):
                return {
                    "focal_predictor": self.df[self.config.focal_predictor]
                    .iloc[idx]
                    .values.astype(self.dtype),
                    "controlled_predictors": self.df[self.config.controlled_predictors]
                    .iloc[idx]
                    .values.astype(self.dtype),
                    "moderators": [
                        self.df[i_p].iloc[idx].values.astype(self.dtype)
                        for i_p in self.config.moderators
                    ],
                    "outcome": self.df[self.config.outcome]
                    .iloc[idx]
                    .values.astype(self.dtype),
                    **(
                        {
                            "weights": self.df[self.config.survey_weights]
                            .iloc[idx]
                            .values.astype(self.dtype)
                        }
                        if self.config.survey_weights is not None
                        else {}
                    ),
                }
            else:
                return {
                    "focal_predictor": self.df[self.config.focal_predictor]
                    .iloc[idx]
                    .values.astype(self.dtype),
                    "controlled_predictors": self.df[self.config.controlled_predictors]
                    .iloc[idx]
                    .values.astype(self.dtype),
                    "moderators": self.df[self.config.moderators]
                    .iloc[idx]
                    .values.astype(self.dtype),
                    "outcome": self.df[self.config.outcome]
                    .iloc[idx]
                    .values.astype(self.dtype),
                    **(
                        {
                            "weights": self.df[self.config.survey_weights]
                            .iloc[idx]
                            .values.astype(self.dtype)
                        }
                        if self.config.survey_weights is not None
                        else {}
                    ),
                }

    def __repr__(self):
        return f"ReGNNDataset with {len(self)} samples"

    def get_subset(self, indices: Sequence[int]) -> "ReGNNDataset":
        dataset_subset = ReGNNDataset(
            self.df_orig.iloc[indices],
            config=self.config,
            output_mode=self.output_mode,
            device=self.device,
            dtype=self.dtype,
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
