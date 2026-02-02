"""Integration tests for ReGNN training with actual DataLoader and backward passes."""
import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from regnn.data.dataset import ReGNNDataset
from regnn.data.base import ReGNNDatasetConfig
from regnn.model.base import ReGNNConfig
from regnn.model.regnn import ReGNN


@pytest.fixture
def synthetic_data():
    """Create synthetic data for testing."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame({
        "focal_predictor": np.random.normal(0, 1, n_samples),
        "control1": np.random.normal(0, 1, n_samples),
        "moderator1": np.random.normal(0, 1, n_samples),
        "moderator2": np.random.normal(0, 1, n_samples),
        "outcome": np.random.normal(0, 1, n_samples),
    })


@pytest.fixture
def regnn_dataset(synthetic_data):
    """Create ReGNN dataset."""
    config = ReGNNDatasetConfig(
        focal_predictor="focal_predictor",
        controlled_predictors=["control1"],
        moderators=["moderator1", "moderator2"],
        outcome="outcome",
        survey_weights=None,
        rename_dict={},
        df_dtypes={},
        preprocess_steps=[],
    )
    
    return ReGNNDataset(
        df=synthetic_data,
        config=config,
        output_mode="tensor",
        device="cpu",
        dtype=np.float32,
    )


@pytest.mark.parametrize("use_soft_tree,tree_depth,vae,batch_norm", [
    (False, None, True, True),   # MLP with VAE and batch norm
    (False, None, False, True),  # MLP without VAE, with batch norm
    (False, None, False, False), # MLP without VAE or batch norm
    (True, 3, False, False),     # SoftTree (VAE not supported)
    (True, 3, False, True),      # SoftTree with batch norm
])
def test_training_backward_pass(regnn_dataset, use_soft_tree, tree_depth, vae, batch_norm):
    """Test full training loop with backward pass for different configurations."""
    
    # Enable anomaly detection for SoftTree tests to find in-place operations
    if use_soft_tree:
        torch.autograd.set_detect_anomaly(True)
    
    # Create model config
    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8, 4] if not use_soft_tree else [8],
        use_soft_tree=use_soft_tree,
        tree_depth=tree_depth,
        vae=vae,
        batch_norm=batch_norm,
        dropout=0.0,
        device="cpu",
        output_mu_var=False,
    )
    
    model = ReGNN.from_config(model_config)
    model.train()
    
    # Create DataLoader
    dataloader = DataLoader(
        regnn_dataset,
        batch_size=16,
        shuffle=True,
    )
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # Training iteration
    for batch_idx, batch_data in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Prepare model inputs
        model_inputs = {
            "moderators": batch_data["moderators"],
            "focal_predictor": batch_data["focal_predictor"],
            "controlled_predictors": batch_data["controlled_predictors"],
        }
        targets = batch_data["outcome"]
        
        # Forward pass
        if vae and model.training:
            # VAE returns (prediction, mu, logvar) in training mode with output_mu_var=False
            predictions = model(**model_inputs)
        else:
            predictions = model(**model_inputs)
        
        # Loss computation
        loss = loss_fn(predictions, targets)
        
        # Backward pass - THIS IS WHERE IN-PLACE ERRORS OCCUR
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Only test first batch
        break
    
    # If we get here without RuntimeError, test passes
    assert True


def test_training_with_3d_inputs(regnn_dataset):
    """Test that model handles 3D inputs from DataLoader correctly."""
    
    # Create model
    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8],
        use_soft_tree=True,
        tree_depth=3,
        vae=False,
        batch_norm=False,
        device="cpu",
    )
    
    model = ReGNN.from_config(model_config)
    model.train()
    
    # Create DataLoader
    dataloader = DataLoader(regnn_dataset, batch_size=16)
    
    # Get one batch
    batch_data = next(iter(dataloader))
    
    # Manually add extra dimensions to simulate potential 3D input issue
    model_inputs = {
        "moderators": batch_data["moderators"].unsqueeze(1),  # Make 3D
        "focal_predictor": batch_data["focal_predictor"].unsqueeze(1),  # Make 2D/3D
        "controlled_predictors": batch_data["controlled_predictors"].unsqueeze(1),  # Make 3D
    }
    
    # Forward pass should handle 3D inputs
    predictions = model(**model_inputs)
    
    # Check output shape is 2D
    assert predictions.ndim == 2
    assert predictions.shape == (16, 1)


def test_training_shape_consistency(regnn_dataset):
    """Test that model output matches target shape through full training loop."""
    
    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8],
        use_soft_tree=True,
        tree_depth=3,
        vae=False,
        batch_norm=True,
        device="cpu",
    )
    
    model = ReGNN.from_config(model_config)
    model.train()
    
    dataloader = DataLoader(regnn_dataset, batch_size=32, shuffle=False)
    
    for batch_data in dataloader:
        model_inputs = {
            "moderators": batch_data["moderators"],
            "focal_predictor": batch_data["focal_predictor"],
            "controlled_predictors": batch_data["controlled_predictors"],
        }
        targets = batch_data["outcome"]
        
        predictions = model(**model_inputs)
        
        # Check shapes match for loss computation
        assert predictions.shape[0] == targets.shape[0], \
            f"Batch size mismatch: predictions {predictions.shape} vs targets {targets.shape}"
        
        # Predictions should be 2D: (batch, 1)
        assert predictions.ndim == 2, \
            f"Predictions should be 2D, got {predictions.ndim}D with shape {predictions.shape}"
        
        # Can compute loss without shape mismatch
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, targets)
        
        # Loss should be scalar
        assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
        
        break  # Test first batch only


def test_multiple_epochs_no_error(regnn_dataset):
    """Test multiple epochs to ensure no accumulated errors."""
    
    model_config = ReGNNConfig.create(
        num_moderators=2,
        num_controlled=1,
        layer_input_sizes=[8],
        use_soft_tree=True,
        tree_depth=3,
        vae=False,
        batch_norm=True,
        device="cpu",
    )
    
    model = ReGNN.from_config(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    dataloader = DataLoader(regnn_dataset, batch_size=16, shuffle=True)
    
    # Run multiple epochs
    for epoch in range(3):
        model.train()
        epoch_loss = 0.0
        
        for batch_data in dataloader:
            optimizer.zero_grad()
            
            model_inputs = {
                "moderators": batch_data["moderators"],
                "focal_predictor": batch_data["focal_predictor"],
                "controlled_predictors": batch_data["controlled_predictors"],
            }
            targets = batch_data["outcome"]
            
            predictions = model(**model_inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Should complete without errors
        assert epoch_loss > 0
