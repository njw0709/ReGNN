import pytest
import pandas as pd
import numpy as np
from regnn.data.preprocess_fns import (
    binary_to_one_hot,
    multi_cat_to_one_hot,
    standardize_cols,
    convert_categorical_to_ordinal,
    map_to_zero_one,
)


@pytest.fixture
def sample_df():
    # Create a sample DataFrame with various data types
    df = pd.DataFrame(
        {
            "binary": pd.Series([True, False, True, False], dtype="category"),
            "multi_cat": pd.Series(["A", "B", "C", "A"], dtype="category"),
            "numeric": [1.0, 2.0, 3.0, 4.0],
            "ordinal": pd.Series(["Low", "Medium", "High", "Low"], dtype="category"),
        }
    )
    return df


def test_binary_to_one_hot_reverse(sample_df):
    # Test binary to one-hot encoding and reverse
    df_orig = sample_df.copy()
    df_processed, cols = binary_to_one_hot(sample_df.copy(), ["binary"])
    df_reversed, _ = binary_to_one_hot._reverse_transform(df_processed, ["binary"])

    # Check if reversed data matches original
    pd.testing.assert_series_equal(
        df_orig["binary"], df_reversed["binary"], check_dtype=True
    )


def test_multi_cat_to_one_hot_reverse(sample_df):
    # Test multi-category to one-hot encoding and reverse
    df_orig = sample_df.copy()
    df_processed, new_cols = multi_cat_to_one_hot(sample_df.copy(), ["multi_cat"])
    df_reversed, _ = multi_cat_to_one_hot._reverse_transform(
        df_processed, ["multi_cat"]
    )

    # Check if reversed data matches original
    pd.testing.assert_series_equal(
        df_orig["multi_cat"], df_reversed["multi_cat"], check_dtype=True
    )


def test_standardize_cols_reverse(sample_df):
    # Test standardization and reverse
    df_orig = sample_df.copy()
    df_processed, mean_std_dict = standardize_cols(sample_df.copy(), ["numeric"])
    df_reversed, _ = standardize_cols._reverse_transform(
        df_processed, ["numeric"], mean_std_dict
    )

    # Check if reversed data matches original
    pd.testing.assert_series_equal(
        df_orig["numeric"],
        df_reversed["numeric"],
        check_dtype=False,  # Float precision might differ slightly
    )


def test_convert_categorical_to_ordinal_reverse(sample_df):
    # Test ordinal conversion and reverse
    df_orig = sample_df.copy()
    df_processed, cols = convert_categorical_to_ordinal(sample_df.copy(), ["ordinal"])
    df_reversed, _ = convert_categorical_to_ordinal._reverse_transform(
        df_processed, ["ordinal"]
    )

    # Check if reversed data matches original
    pd.testing.assert_series_equal(
        df_orig["ordinal"], df_reversed["ordinal"], check_dtype=True
    )


def test_map_to_zero_one_reverse(sample_df):
    # Test min-max scaling and reverse
    df_orig = sample_df.copy()
    df_processed, cols = map_to_zero_one(sample_df.copy(), ["numeric"])

    # Create min_max_dict from original data
    min_max_dict = {"numeric": (df_orig["numeric"].min(), df_orig["numeric"].max())}

    df_reversed, _ = map_to_zero_one._reverse_transform(
        df_processed, ["numeric"], min_max_dict
    )

    # Check if reversed data matches original
    pd.testing.assert_series_equal(
        df_orig["numeric"],
        df_reversed["numeric"],
        check_dtype=False,  # Float precision might differ slightly
    )


def test_combined_transformations(sample_df):
    # Test multiple transformations in sequence
    df_orig = sample_df.copy()

    # Apply transformations
    df_processed, mean_std_dict = standardize_cols(sample_df.copy(), ["numeric"])
    df_processed, cols = map_to_zero_one(df_processed, ["numeric"])

    # Reverse transformations in opposite order
    min_max_dict = {"numeric": (df_orig["numeric"].min(), df_orig["numeric"].max())}
    df_reversed, _ = map_to_zero_one._reverse_transform(
        df_processed, ["numeric"], min_max_dict
    )
    df_reversed, _ = standardize_cols._reverse_transform(
        df_reversed, ["numeric"], mean_std_dict
    )

    # Check if reversed data matches original
    pd.testing.assert_series_equal(
        df_orig["numeric"],
        df_reversed["numeric"],
        check_dtype=False,  # Float precision might differ slightly
    )


def test_edge_cases():
    # Test with empty DataFrame
    df_empty = pd.DataFrame()
    df_processed, cols = standardize_cols(df_empty.copy(), None)
    df_reversed, _ = standardize_cols._reverse_transform(df_processed, None, {})
    assert df_reversed.empty

    # Test with single value
    df_single = pd.DataFrame({"numeric": [1.0]})
    df_processed, mean_std_dict = standardize_cols(df_single.copy(), ["numeric"])
    df_reversed, _ = standardize_cols._reverse_transform(
        df_processed, ["numeric"], mean_std_dict
    )
    pd.testing.assert_series_equal(
        df_single["numeric"], df_reversed["numeric"], check_dtype=False
    )
