import pytest
import pandas as pd
import numpy as np
import torch
from regnn.data.dataset import ReGNNDataset


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
    return ReGNNDataset(
        df=sample_df,
        focal_predictor="focal_predictor",
        controlled_predictors=["control1", "control2"],
        moderators=["moderator1", "moderator2"],
        outcome="outcome",
        survey_weights="weights",
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

    def simple_standardize(df, columns):
        """Simple standardization function for testing"""
        mean_std_dict = {}
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
            mean_std_dict[col] = (mean, std)
        return df, mean_std_dict

    # Standardize
    dataset.standardize([(["control1", "control2"], simple_standardize)])

    # Check if mean_std_dict is populated
    assert "control1" in dataset.mean_std_dict
    assert "control2" in dataset.mean_std_dict

    # Get standardized values
    std_values = dataset.df[["control1", "control2"]].copy()

    # Reverse standardize
    dataset.reverse_standardize(["control1", "control2"])

    # Check if values are back to original (approximately)
    original_mean1 = dataset.df["control1"].mean()
    original_std1 = dataset.df["control1"].std()
    assert abs(original_mean1) > 0.01  # Not close to 0 anymore
    assert abs(original_std1 - 1.0) > 0.01  # Not close to 1 anymore


def test_list_moderators(sample_df):
    """Test with list of lists for moderators"""
    dataset = ReGNNDataset(
        df=sample_df,
        focal_predictor="focal_predictor",
        controlled_predictors=["control1", "control2"],
        moderators=[["moderator1"], ["moderator2"]],
        outcome="outcome",
        survey_weights="weights",
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
