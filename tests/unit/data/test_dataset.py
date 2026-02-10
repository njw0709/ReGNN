import pytest
import pandas as pd
import numpy as np
import torch
from regnn.data.dataset import ReGNNDataset
from regnn.data.base import ReGNNDatasetConfig, PreprocessStep
from regnn.data.preprocess_fns import (
    standardize_cols,
    multi_cat_to_one_hot,
    binary_to_one_hot,
    debias_focal_predictor,
)
from sklearn.linear_model import Ridge


@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing"""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "focal_predictor": np.random.normal(0, 1, 100),
            "control1": np.random.normal(0, 1, 100),
            "control2": np.random.normal(0, 1, 100),
            "moderator1": np.random.normal(0, 1, 100),
            "moderator2": np.random.normal(0, 1, 100),
            "outcome": np.random.normal(0, 1, 100),
            "weights": np.random.uniform(0.5, 1.5, 100),
        }
    )
    return df


@pytest.fixture
def dataset(sample_df):
    """Create a ReGNNDataset instance"""
    config = ReGNNDatasetConfig(
        focal_predictor="focal_predictor",
        controlled_predictors=["control1", "control2"],
        moderators=["moderator1", "moderator2"],
        outcome="outcome",
        survey_weights="weights",
        rename_dict={},
        df_dtypes={},
        preprocess_steps=[
            PreprocessStep(
                columns=["control1", "control2"],
                function=standardize_cols,
            ),
        ],
    )
    return ReGNNDataset(
        df=sample_df,
        config=config,
    )


def test_dataset_init(dataset, sample_df):
    """Test dataset initialization"""
    assert len(dataset) == len(sample_df)
    assert dataset.config.focal_predictor == "focal_predictor"
    assert dataset.config.controlled_predictors == ["control1", "control2"]
    assert dataset.config.moderators == ["moderator1", "moderator2"]
    assert dataset.config.outcome == "outcome"
    assert dataset.config.survey_weights == "weights"


def test_dataset_getitem(dataset):
    """Test __getitem__ method"""
    item = dataset[0]
    assert "focal_predictor" in item
    assert "controlled" in item
    assert "moderators" in item
    assert "outcome" in item
    assert "weights" in item

    # Check shapes
    assert item["focal_predictor"].shape == (1,)
    assert item["controlled"].shape == (2,)
    assert item["moderators"].shape == (2,)
    assert item["outcome"].shape == (1,)
    assert item["weights"].shape == (1,)


def test_dataset_to_numpy(dataset):
    """Test to_numpy method"""
    numpy_data = dataset.to_numpy()
    assert "focal_predictor" in numpy_data
    assert "controlled_predictors" in numpy_data
    assert "moderators" in numpy_data
    assert "outcome" in numpy_data
    assert "weights" in numpy_data

    # Check shapes
    assert numpy_data["focal_predictor"].shape == (100,)
    assert numpy_data["controlled_predictors"].shape == (100, 2)
    assert numpy_data["moderators"].shape == (100, 2)
    assert numpy_data["outcome"].shape == (100,)
    assert numpy_data["weights"].shape == (100,)


def test_dataset_to_tensor(dataset):
    """Test to_tensor method"""
    tensor_data = dataset.to_tensor()
    assert "focal_predictor" in tensor_data
    assert "controlled_predictors" in tensor_data
    assert "moderators" in tensor_data
    assert "outcome" in tensor_data
    assert "weights" in tensor_data

    # Check types and shapes
    assert isinstance(tensor_data["focal_predictor"], torch.Tensor)
    assert tensor_data["focal_predictor"].shape == (100,)
    assert tensor_data["controlled_predictors"].shape == (100, 2)
    assert tensor_data["moderators"].shape == (100, 2)
    assert tensor_data["outcome"].shape == (100,)
    assert tensor_data["weights"].shape == (100,)


def test_dataset_get_subset(dataset):
    """Test get_subset method"""
    subset = dataset.get_subset([0, 1, 2, 3, 4])
    assert len(subset) == 5
    assert subset.config.focal_predictor == dataset.config.focal_predictor
    assert subset.config.controlled_predictors == dataset.config.controlled_predictors
    assert subset.config.moderators == dataset.config.moderators


def test_to_torch_dataset(dataset):
    """Test to_torch_dataset method"""
    torch_dataset = dataset.to_torch_dataset()
    assert len(torch_dataset) == len(dataset)

    # Test getting an item
    item = torch_dataset[0]
    assert "focal_predictor" in item
    assert "controlled_predictors" in item
    assert "moderators" in item
    assert "outcome" in item
    assert "weights" in item

    # Check types
    assert isinstance(item["focal_predictor"], torch.Tensor)
    assert isinstance(item["controlled_predictors"], torch.Tensor)
    assert isinstance(item["moderators"], torch.Tensor)
    assert isinstance(item["outcome"], torch.Tensor)
    assert isinstance(item["weights"], torch.Tensor)


def test_standardize_and_reverse(dataset):
    """Test standardization and reverse standardization"""
    # Store original values
    original_values = dataset.df[["control1", "control2"]].copy()

    # Apply preprocessing from config
    dataset.preprocess(inplace=True)

    # Check if standardized
    std_values = dataset.df[["control1", "control2"]]
    assert abs(std_values["control1"].mean()) < 0.01  # Close to 0
    assert abs(std_values["control1"].std() - 1.0) < 0.01  # Close to 1

    # Reverse standardize
    dataset.reverse_preprocess(inplace=True)

    # Check if values are back to original (approximately)
    pd.testing.assert_frame_equal(
        original_values,
        dataset.df[["control1", "control2"]],
        check_dtype=False,
        check_exact=False,
        rtol=1e-5,
        atol=1e-5,
    )


def test_list_moderators(sample_df):
    """Test with list of lists for moderators"""
    config = ReGNNDatasetConfig(
        focal_predictor="focal_predictor",
        controlled_predictors=["control1", "control2"],
        moderators=[["moderator1"], ["moderator2"]],
        outcome="outcome",
        survey_weights="weights",
        rename_dict={},
        df_dtypes={},
        preprocess_steps=[
            PreprocessStep(
                columns=["control1", "control2"],
                function=standardize_cols,
            ),
        ],
    )
    dataset = ReGNNDataset(
        df=sample_df,
        config=config,
    )

    # Test __getitem__
    item = dataset[0]
    assert isinstance(item["moderators"], list)
    assert len(item["moderators"]) == 2

    # Test to_numpy
    numpy_data = dataset.to_numpy()
    assert isinstance(numpy_data["moderators"], list)
    assert len(numpy_data["moderators"]) == 2

    # Test to_tensor
    tensor_data = dataset.to_tensor()
    assert isinstance(tensor_data["moderators"], list)
    assert len(tensor_data["moderators"]) == 2

    # Test to_torch_dataset
    torch_dataset = dataset.to_torch_dataset()
    item = torch_dataset[0]
    assert isinstance(item["moderators"], list)
    assert len(item["moderators"]) == 2


def test_preprocessing_multiple_steps(sample_df):
    """Test multiple preprocessing steps and their reversal"""
    # Create categorical columns for testing
    sample_df["cat_col"] = pd.Series(["A", "B", "C"] * 33 + ["A"], dtype="category")
    sample_df["binary_col"] = pd.Series([True, False] * 50, dtype="category")

    config = ReGNNDatasetConfig(
        focal_predictor="focal_predictor",
        controlled_predictors=["control1", "control2", "cat_col", "binary_col"],
        moderators=["moderator1", "moderator2"],
        outcome="outcome",
        survey_weights="weights",
        rename_dict={},
        df_dtypes={"category": ["cat_col"], "binary": ["binary_col"]},
        preprocess_steps=[
            PreprocessStep(
                columns=["control1", "control2"],
                function=standardize_cols,
            ),
            PreprocessStep(
                columns=["cat_col"],
                function=multi_cat_to_one_hot,
            ),
            PreprocessStep(
                columns=["binary_col"],
                function=binary_to_one_hot,
            ),
        ],
    )

    # Create dataset and store original data
    dataset = ReGNNDataset(df=sample_df.copy(), config=config)
    original_data = dataset.df.copy()

    # Apply preprocessing
    dataset.preprocess(inplace=True)

    # Check standardization
    assert abs(dataset.df["control1"].mean()) < 0.01
    assert abs(dataset.df["control1"].std() - 1.0) < 0.01
    assert abs(dataset.df["control2"].mean()) < 0.01
    assert abs(dataset.df["control2"].std() - 1.0) < 0.01

    # Check one-hot encoding
    assert "cat_col_B" in dataset.df.columns
    assert "cat_col_C" in dataset.df.columns
    assert "binary_col" in dataset.df.columns

    # Verify df_orig contains original data
    pd.testing.assert_frame_equal(dataset.df_orig, sample_df)

    # Reverse preprocessing
    dataset.reverse_preprocess(inplace=True)

    # Check if data is back to original
    pd.testing.assert_frame_equal(
        original_data[["control1", "control2"]],
        dataset.df[["control1", "control2"]],
        check_dtype=False,
        check_exact=False,
        rtol=1e-5,
        atol=1e-5,
    )

    # Check categorical columns are restored correctly
    pd.testing.assert_series_equal(
        original_data["cat_col"],
        dataset.df["cat_col"],
        check_dtype=True,
        check_categorical=True,
    )
    pd.testing.assert_series_equal(
        original_data["binary_col"],
        dataset.df["binary_col"],
        check_dtype=True,
        check_categorical=True,
    )


def test_preprocessing_subset_preservation(sample_df):
    """Test that preprocessing information is preserved when getting subset"""
    config = ReGNNDatasetConfig(
        focal_predictor="focal_predictor",
        controlled_predictors=["control1", "control2"],
        moderators=["moderator1", "moderator2"],
        outcome="outcome",
        survey_weights="weights",
        rename_dict={},
        df_dtypes={},
        preprocess_steps=[
            PreprocessStep(
                columns=["control1", "control2"],
                function=standardize_cols,
            ),
        ],
    )

    # Create dataset and preprocess
    dataset = ReGNNDataset(df=sample_df.copy(), config=config)
    dataset.preprocess(inplace=True)

    # Get subset
    subset = dataset.get_subset([0, 1, 2, 3, 4])

    # Verify subset has same preprocessing steps
    assert len(subset.config.preprocess_steps) == len(dataset.config.preprocess_steps)

    # Verify subset can be reversed correctly
    original_subset = subset.df.copy()
    subset.reverse_preprocess(inplace=True)

    # Check if reversed data matches original subset
    pd.testing.assert_frame_equal(
        dataset.df_orig.iloc[[0, 1, 2, 3, 4]][["control1", "control2"]],
        subset.df[["control1", "control2"]],
        check_dtype=False,
        check_exact=False,
        rtol=1e-5,
        atol=1e-5,
    )


def test_df_orig_independence(sample_df):
    """Test that df_orig remains unchanged through preprocessing operations"""
    config = ReGNNDatasetConfig(
        focal_predictor="focal_predictor",
        controlled_predictors=["control1", "control2"],
        moderators=["moderator1", "moderator2"],
        outcome="outcome",
        survey_weights="weights",
        rename_dict={},
        df_dtypes={},
        preprocess_steps=[
            PreprocessStep(
                columns=["control1", "control2"],
                function=standardize_cols,
            ),
        ],
    )

    # Create dataset and store original data
    dataset = ReGNNDataset(df=sample_df.copy(), config=config)
    original_df = sample_df.copy()

    # Multiple preprocessing and reverse preprocessing operations
    for _ in range(3):
        dataset.preprocess(inplace=True)
        dataset.reverse_preprocess(inplace=True)

    # Verify df_orig remained unchanged
    pd.testing.assert_frame_equal(dataset.df_orig, original_df)


def test_df_orig_updated_with_debiasing(sample_df):
    """Test that df_orig is updated with residualized focal predictor when debiasing is applied"""
    # Create config with debias preprocessing step
    config = ReGNNDatasetConfig(
        focal_predictor="focal_predictor",
        controlled_predictors=["control1", "control2"],
        moderators=["moderator1", "moderator2"],
        outcome="outcome",
        survey_weights="weights",
        rename_dict={},
        df_dtypes={},
        preprocess_steps=[
            PreprocessStep(
                columns=["focal_predictor"],
                function=debias_focal_predictor,
                reverse_transform_info={
                    "controlled_predictors": ["control1", "control2"],
                    "model_class": Ridge,
                    "k": 3,
                    "is_classifier": False,
                },
            ),
        ],
    )

    # Create dataset and store original focal predictor values
    dataset = ReGNNDataset(df=sample_df.copy(), config=config)
    original_focal_values = dataset.df_orig["focal_predictor"].copy()

    # Apply preprocessing (which includes debiasing)
    dataset.preprocess(inplace=True)

    # Verify that df_orig focal predictor has been updated (residualized)
    # The residualized values should be different from the original
    assert not np.allclose(
        dataset.df_orig["focal_predictor"].values, original_focal_values.values
    ), "df_orig focal predictor should be updated with residualized values"

    # Verify that df and df_orig have the same focal predictor values after debiasing
    np.testing.assert_array_equal(
        dataset.df["focal_predictor"].values,
        dataset.df_orig["focal_predictor"].values,
        err_msg="df and df_orig should have the same residualized focal predictor values",
    )

    # Verify that other columns in df_orig remain unchanged
    for col in ["control1", "control2", "moderator1", "moderator2", "outcome", "weights"]:
        np.testing.assert_array_equal(
            dataset.df_orig[col].values,
            sample_df[col].values,
            err_msg=f"Column {col} in df_orig should remain unchanged",
        )


def test_df_orig_independence_without_debiasing(sample_df):
    """Test that df_orig remains unchanged when debiasing is NOT applied"""
    config = ReGNNDatasetConfig(
        focal_predictor="focal_predictor",
        controlled_predictors=["control1", "control2"],
        moderators=["moderator1", "moderator2"],
        outcome="outcome",
        survey_weights="weights",
        rename_dict={},
        df_dtypes={},
        preprocess_steps=[
            PreprocessStep(
                columns=["control1", "control2"],
                function=standardize_cols,
            ),
        ],
    )

    # Create dataset and store original data
    dataset = ReGNNDataset(df=sample_df.copy(), config=config)
    original_df = sample_df.copy()

    # Apply preprocessing (no debiasing)
    dataset.preprocess(inplace=True)

    # Verify df_orig remained unchanged
    pd.testing.assert_frame_equal(dataset.df_orig, original_df)
