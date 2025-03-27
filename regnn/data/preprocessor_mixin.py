from typing import List, Optional
import pandas as pd
from .base import PreprocessStep
from .preprocess_fns import (
    standardize_cols,
    map_to_zero_one,
    multi_cat_to_one_hot,
    binary_to_one_hot,
)


class PreprocessorMixin:
    """Mixin for preprocessing operations"""

    def preprocess(
        self,
        inplace: bool = True,
    ):
        """Apply preprocessing steps defined in config.preprocess_steps"""
        df_temp = self.df.copy()
        for step in self.config.preprocess_steps:
            # Store original columns for reverse transformation
            original_cols = step.columns.copy()

            # Apply preprocessing
            df_temp, return_value = step.function(df_temp, step.columns)

            # Update column lists if columns were changed
            if isinstance(return_value, dict):
                # For functions that return a categories dictionary (like multi_cat_to_one_hot)
                new_colnames = [
                    f"{col}_{cat}"
                    for col in step.columns
                    for cat in return_value[col][1:]
                ]  # Exclude first category
            else:
                # For functions that return column names
                new_colnames = return_value

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

                # Store the return value directly as reverse_transform_info
                if step.function == standardize_cols:
                    step.reverse_transform_info = {"mean_std_dict": return_value}
                elif step.function == map_to_zero_one:
                    step.reverse_transform_info = {"min_max_dict": return_value}
                elif step.function == multi_cat_to_one_hot:
                    step.reverse_transform_info = {"categories_dict": return_value}
                elif step.function == binary_to_one_hot:
                    # binary_to_one_hot doesn't need additional info
                    step.reverse_transform_info = {}
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
        df: Optional[pd.DataFrame] = None,
        inplace: bool = True,
    ) -> pd.DataFrame:
        """
        Reverse all preprocessing steps in reverse order using steps from config.

        Args:
            df: DataFrame to reverse preprocess. If None, uses self.df
            inplace: If True, modifies self.df. If False, returns new DataFrame

        Returns:
            pd.DataFrame: Reversed preprocessed DataFrame
        """
        df_temp = self.df.copy() if df is None else df.copy()

        # Reverse the preprocessing steps in reverse order
        for step in reversed(self.config.preprocess_steps):
            # For functions with built-in reverse transform
            df_temp, _ = step.reverse_function(
                df_temp, step.columns, **step.reverse_transform_info
            )
        if inplace:
            self.df = df_temp
            return df_temp
        else:
            return df_temp
