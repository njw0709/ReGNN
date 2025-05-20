import pytest
import pandas as pd
import numpy as np
import torch
from regnn.data.dataset import ReGNNDataset
from regnn.train.base import TrainingConfig
from regnn.train.loop import train_iteration
from regnn.macroutils.utils import get_l2_length, get_gradient_norms
from regnn.model.regnn import ReGNN
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


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
        df_dtypes={},
        rename_dict={},
    )


@pytest.fixture
def model():
    """Create a ReGNN model instance"""
    return ReGNN(
        num_moderators=2,
        num_controlled=2,
        hidden_layer_sizes=[10, 5, 1],
        include_bias_focal_predictor=True,
        control_moderators=True,
        batch_norm=True,
        vae=False,
        device="cpu",
    )


@pytest.fixture
def training_config():
    """Create a training configuration"""
    return TrainingConfig(
        hidden_layer_sizes=[10, 5, 1], epochs=2, batch_size=16, lr=0.001, device="cpu"
    )


def test_training_config():
    """Test TrainingConfig initialization"""
    config = TrainingConfig(hidden_layer_sizes=[10, 5, 1])
    assert config.hidden_layer_sizes == [10, 5, 1]
    assert config.epochs == 100  # Default value
    assert config.device == "cuda" if torch.cuda.is_available() else "cpu"


def test_get_l2_length(model):
    """Test get_l2_length function"""
    l2_lengths = get_l2_length(model)
    assert "main" in l2_lengths
    assert "index" in l2_lengths
    assert isinstance(l2_lengths["main"], float)
    assert isinstance(l2_lengths["index"], float)


def test_get_gradient_norms(model):
    """Test get_gradient_norms function"""
    # Create dummy input and perform a forward and backward pass
    x_moderators = torch.randn(2, 2)
    x_focal = torch.randn(2)
    x_controlled = torch.randn(2, 2)
    y = torch.randn(2, 1)

    # Set model to eval mode to avoid batch normalization issues with small batches
    model.eval()

    output = model(x_moderators, x_focal, x_controlled)
    loss = nn.MSELoss()(output, y)
    loss.backward()

    grad_norms = get_gradient_norms(model)
    assert "main" in grad_norms
    assert "index" in grad_norms
    assert isinstance(grad_norms["main"], list)
    assert isinstance(grad_norms["index"], list)


def test_train_iteration(model, dataset):
    """Test train_iteration function"""
    # Create dataloader
    torch_dataset = dataset.to_torch_dataset(device="cpu")
    dataloader = DataLoader(torch_dataset, batch_size=16, shuffle=True)

    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss(reduction="none")

    # Run one training iteration
    epoch_loss, l2_lengths = train_iteration(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
        get_l2_lengths=True,
    )

    assert isinstance(epoch_loss, float)
    assert isinstance(l2_lengths, list)
    assert len(l2_lengths) > 0
    assert "main" in l2_lengths[0]
    assert "index" in l2_lengths[0]
