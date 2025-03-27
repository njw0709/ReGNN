from typing import Sequence, Callable, Union, Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from .base import numeric, PreprocessStep
from .preprocess_fns import standardize_cols, map_to_zero_one, multi_cat_to_one_hot


class PreprocessorMixin:
    """Mixin for preprocessing operations"""

    def preprocess(
        self,
        preprocess_list: List[PreprocessStep],
        inplace: bool = True,
    ):
        df_temp = self.df.copy()
        for step in preprocess_list:
            # Store original columns for reverse transformation
            original_cols = step.columns.copy()

            # Apply preprocessing
            df_temp, new_colnames = step.function(df_temp, step.columns)

            # Update column lists if columns were changed
            if set(new_colnames) != set(step.columns):
                for c in step.columns:
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

            # Store reverse transformation info
            if hasattr(step.function, "_reverse_transform"):
                # For functions that have built-in reverse transform
                step.reverse_function = step.function._reverse_transform
                # Store any additional info needed for reverse transform
                if step.function == standardize_cols:
                    step.reverse_transform_info = {"mean_std_dict": new_colnames}
                elif step.function == map_to_zero_one:
                    # Get min_max_dict from the function's return value
                    _, min_max_dict = step.function(df_temp.copy(), step.columns)
                    step.reverse_transform_info = {"min_max_dict": min_max_dict}
                elif step.function == multi_cat_to_one_hot:
                    # Create mapping of original columns to their new one-hot columns
                    column_mapping = {}
                    for orig_col in step.columns:
                        column_mapping[orig_col] = [
                            col
                            for col in new_colnames
                            if col.startswith(f"{orig_col}_")
                        ]
                    step.reverse_transform_info = {"column_mapping": column_mapping}
                else:
                    step.reverse_transform_info = {}
            else:
                # For functions without built-in reverse transform
                step.reverse_transform_info = {
                    "original_columns": original_cols,
                    "new_columns": new_colnames,
                }

        if inplace:
            self.df = df_temp
            return df_temp
        else:
            return df_temp

    def reverse_preprocess(
        self,
        preprocess_list: List[PreprocessStep],
        df: Optional[pd.DataFrame] = None,
        inplace: bool = True,
    ) -> pd.DataFrame:
        """
        Reverse all preprocessing steps in reverse order.

        Args:
            preprocess_list: List of PreprocessStep objects that were used in preprocessing
            df: DataFrame to reverse preprocess. If None, uses self.df
            inplace: If True, modifies self.df. If False, returns new DataFrame

        Returns:
            pd.DataFrame: Reversed preprocessed DataFrame
        """
        df_temp = self.df.copy() if df is None else df.copy()

        # Reverse the preprocessing steps in reverse order
        for step in reversed(preprocess_list):
            # For functions with built-in reverse transform
            df_temp, _ = step.reverse_function(
                df_temp, step.columns, **step.reverse_transform_info
            )
        if inplace:
            self.df = df_temp
            return df_temp
        else:
            return df_temp
