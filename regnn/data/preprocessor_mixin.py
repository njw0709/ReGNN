from typing import Optional
import pandas as pd


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
                if "category_map" in return_value.keys():
                    new_colnames = [
                        f"{col}_{cat}"
                        for col in step.columns
                        for cat in return_value["category_map"][col][1:]
                    ]  # Exclude first category
                else:
                    new_colnames = step.columns
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
                if isinstance(return_value, dict):
                    step.reverse_transform_info = return_value
                else:
                    step.reverse_transform_info = {
                        "original_columns": original_cols,
                        "new_columns": new_colnames,
                    }
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
