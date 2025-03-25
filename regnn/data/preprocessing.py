from typing import Sequence, Callable, Union, Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from .base import BaseDataset, PreprocessStep, numeric


class PreprocessingMixin:
    """Mixin for preprocessing operations"""

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
                    lists_to_check = [
                        self.config.controlled_predictors,
                        self.config.moderators,
                    ]
                    for current_list in lists_to_check:
                        if c in current_list:
                            current_list.remove(c)
                            current_list.extend(
                                [new_c for new_c in new_colnames if c in new_c]
                            )
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
