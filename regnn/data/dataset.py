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

    def _to_tensor(self, value, dtype=None) -> torch.Tensor:
        """Convert a value to a tensor with proper shape."""
        if dtype is None:
            dtype = self.dtype

        if not isinstance(dtype, torch.dtype):
            try:
                # np.dtype() is used to ensure 'dtype' is a valid numpy dtype identifier
                # (e.g., np.float32, 'float32', etc.) before creating a dummy array.
                np_dtype_obj = np.dtype(dtype)
                # Create a dummy NumPy array with that dtype, then convert to a tensor
                # to obtain the corresponding torch.dtype.
                dummy_np_array = np.array([], dtype=np_dtype_obj)
                dtype = torch.from_numpy(dummy_np_array).dtype
            except TypeError:
                # This typically occurs if np.dtype(dtype) fails because 'dtype' is not
                # a recognizable format (e.g., invalid string).
                raise ValueError(
                    f"Invalid dtype '{dtype}' provided. Cannot determine a corresponding NumPy dtype "
                    f"for conversion to torch.dtype."
                )
            except Exception as e:
                # Catch any other unexpected errors during the conversion process.
                raise ValueError(
                    f"Could not convert dtype '{dtype}' to torch.dtype. Original error: {e}"
                )
        return torch.tensor(np.array(value), dtype=dtype).to(self.device)

    def _to_numpy(self, value, dtype=None) -> np.ndarray:
        """Convert a value to a numpy array with proper shape."""
        if dtype is None:
            dtype = self.dtype

        if isinstance(dtype, torch.dtype):
            try:
                # Convert torch.dtype to numpy.dtype
                # Create a dummy tensor, convert to numpy, and get its dtype
                dummy_tensor = torch.tensor([], dtype=dtype)
                dtype = dummy_tensor.numpy().dtype
            except Exception as e:
                raise ValueError(
                    f"Could not convert torch.dtype '{dtype}' to numpy.dtype. Original error: {e}"
                )
        elif not isinstance(dtype, np.dtype):
            try:
                # Attempt to interpret as a numpy dtype specifier (e.g., string or type object)
                dtype = np.dtype(dtype)
            except TypeError:
                raise ValueError(
                    f"Invalid dtype '{dtype}' provided. Cannot determine a corresponding NumPy dtype."
                )
        return np.array(value).astype(dtype)

    def _get_item_value(self, key: str, idx: Union[int, slice], as_tensor: bool = True):
        """Get a single item value, either as tensor or numpy array."""
        value = self.df[key].iloc[idx]
        if value.ndim == 0:
            value = [value]
        if isinstance(idx, slice):
            value = np.expand_dims(np.array(value), 1)
        else:
            value = np.expand_dims(np.array(value), 0)
        return self._to_tensor(value) if as_tensor else self._to_numpy(value)

    def _get_moderators(self, idx: int, as_tensor: bool = True):
        """Get moderators, handling both list and non-list cases."""
        if isinstance(self.config.moderators[0], list):
            return [
                self._get_item_value(i_p, idx, as_tensor)
                for i_p in self.config.moderators
            ]
        else:
            return self._get_item_value(self.config.moderators, idx, as_tensor)

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        as_tensor = self.output_mode == "tensor"

        result = {
            "focal_predictor": self._get_item_value(
                self.config.focal_predictor, idx, as_tensor
            ),
            "controlled_predictors": self._get_item_value(
                self.config.controlled_predictors, idx, as_tensor
            ),
            "moderators": self._get_moderators(idx, as_tensor),
            "outcome": self._get_item_value(self.config.outcome, idx, as_tensor),
        }

        if self.config.survey_weights is not None:
            result["weights"] = self._get_item_value(
                self.config.survey_weights, idx, as_tensor
            )

        return result

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
        dataset_subset.df = self.df.iloc[indices]

        # set extra attributes
        all_attrs_new = list(dataset_subset.__dict__.keys())
        for key, value in self.__dict__.items():
            if key not in all_attrs_new:
                setattr(dataset_subset, key, value)
        return dataset_subset

    def dropna(self, inplace: bool = True):
        df = self.df.dropna()
        df_orig = self.df_orig.loc[self.df.index]
        if inplace:
            self.df = df
            self.df_orig = df_orig
        else:
            return df, df_orig

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
