from pydantic import BaseModel, ConfigDict
from numpydantic import NDArray, Shape
from typing import Union, Literal, Dict, List, Optional
import numpy as np
import pandas as pd
from .utils import smart_number_format
from .tests import categorical_test

from scipy.stats import ttest_1samp
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import mannwhitneyu


class Feature(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: NDArray[Union[Shape["n"], Shape["n, 1"], Shape["1, n"]]]
    dtype: Literal["continuous", "binary", "categorical", "ordinal"]

    def compute_summary(self) -> str:
        """Compute summary statistics based on the dtype of the feature.

        Returns:
            str: A formatted string containing relevant summary statistics
        """
        # Ensure data is 1D for easier computation
        data_1d = self.data.reshape(-1)

        if self.dtype == "continuous" or self.dtype == "ordinal":
            return "Mean: {}\nStd: {}".format(
                smart_number_format(np.mean(data_1d)),
                smart_number_format(np.std(data_1d)),
            )

        elif self.dtype == "binary":
            if not np.issubdtype(data_1d.dtype, np.bool_):
                data_1d = data_1d.astype(bool)
            unique, counts = np.unique(data_1d, return_counts=True)
            proportions = counts / len(data_1d)
            return "Prop: {}".format(
                dict(zip(unique, [smart_number_format(p) for p in proportions]))
            )

        elif self.dtype == "categorical":
            unique, counts = np.unique(data_1d, return_counts=True)
            proportions = counts / len(data_1d)
            return "Prop: {}".format(
                dict(zip(unique, [smart_number_format(p) for p in proportions]))
            )

        else:
            raise TypeError("dtype is not what is expected.")

    def test_diff(self, other: "Feature") -> tuple[float, str]:
        """Test if the distribution of this feature is different from another feature.

        Args:
            other: Another Feature instance to compare against

        Returns:
            tuple[float, str]: (p-value, test name)

        Raises:
            ValueError: If dtypes don't match between features
        """
        if self.dtype != other.dtype:
            raise ValueError(
                f"Feature dtypes must match. Got {self.dtype} and {other.dtype}"
            )

        data_1d = self.data.reshape(-1)
        other_1d = other.data.reshape(-1)

        if self.dtype == "continuous":
            # For continuous, use t-test comparing against population mean
            test_result = ttest_1samp(data_1d, popmean=np.mean(other_1d))
            return test_result.pvalue, "t-test"

        elif self.dtype == "ordinal":
            # For ordinal, use Mann-Whitney U test
            test_result = mannwhitneyu(data_1d, other_1d, alternative="two-sided")
            return test_result.pvalue, "mann-whitney-u"

        elif self.dtype == "binary":
            # For binary, use proportions z-test
            if not np.issubdtype(data_1d.dtype, np.bool_):
                data_1d = data_1d.astype(bool)
            if not np.issubdtype(other_1d.dtype, np.bool_):
                other_1d = other_1d.astype(bool)

            count = np.sum(data_1d)
            nobs = len(data_1d)
            pop_prop = np.mean(other_1d)

            test_result = proportions_ztest(count, nobs, value=pop_prop)
            return test_result[1], "proportions z-test"  # [1] is p-value

        elif self.dtype == "categorical":
            # For categorical, use chi-square test
            test_result = categorical_test(data_1d, other_1d)
            return test_result.pvalue, "chi-square test"

        else:
            raise TypeError("dtype is not what is expected.")


class Cluster(BaseModel):
    """A cluster containing multiple features.

    Attributes:
        features: Dictionary mapping feature names to Feature instances
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    features: Dict[str, Feature]

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        continuous_cols: Optional[List[str]] = None,
        binary_cols: Optional[List[str]] = None,
        ordinal_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> "Cluster":
        """Create a Cluster instance from a pandas DataFrame.

        Args:
            df: Input DataFrame
            continuous_cols: List of column names for continuous features
            binary_cols: List of column names for binary features
            ordinal_cols: List of column names for ordinal features
            categorical_cols: List of column names for categorical features

        Returns:
            Cluster: A new Cluster instance

        Raises:
            ValueError: If column lists overlap or contain invalid column names
        """
        # Initialize empty lists if None
        continuous_cols = continuous_cols or []
        binary_cols = binary_cols or []
        ordinal_cols = ordinal_cols or []
        categorical_cols = categorical_cols or []

        # Validate column names
        all_cols = set(continuous_cols + binary_cols + ordinal_cols + categorical_cols)
        invalid_cols = all_cols - set(df.columns)
        if invalid_cols:
            raise ValueError(f"Columns not found in DataFrame: {invalid_cols}")

        # Check for overlapping column lists
        col_lists = [continuous_cols, binary_cols, ordinal_cols, categorical_cols]
        for i, list1 in enumerate(col_lists):
            for list2 in col_lists[i + 1 :]:
                overlap = set(list1) & set(list2)
                if overlap:
                    raise ValueError(f"Column lists overlap: {overlap}")

        # Create features dictionary
        features = {}

        # Add continuous features
        for col in continuous_cols:
            features[col] = Feature(data=df[col].to_numpy(), dtype="continuous")

        # Add binary features
        for col in binary_cols:
            features[col] = Feature(data=df[col].to_numpy(), dtype="binary")

        # Add ordinal features
        for col in ordinal_cols:
            features[col] = Feature(data=df[col].to_numpy(), dtype="ordinal")

        # Add categorical features
        for col in categorical_cols:
            features[col] = Feature(data=df[col].to_numpy(), dtype="categorical")

        return cls(features=features)

    def compute_summary(self) -> Dict[str, str]:
        """Compute summary statistics for all features in the cluster.

        Returns:
            Dict[str, str]: Dictionary mapping feature names to their summary strings
        """
        return {
            name: feature.compute_summary() for name, feature in self.features.items()
        }

    def test_diff(self, other: "Cluster") -> Dict[str, tuple[float, str]]:
        """Test if the distributions of features in this cluster are different from another cluster.

        Args:
            other: Another Cluster instance to compare against

        Returns:
            Dict[str, tuple[float, str]]: Dictionary mapping feature names to (p-value, test name)

        Raises:
            ValueError: If clusters have different features
        """
        if set(self.features.keys()) != set(other.features.keys()):
            raise ValueError(
                "Clusters must have the same features. "
                f"Got {set(self.features.keys())} vs {set(other.features.keys())}"
            )

        return {
            name: self.features[name].test_diff(other.features[name])
            for name in self.features.keys()
        }
