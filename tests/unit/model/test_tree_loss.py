import pytest
import torch
from regnn.train import TreeLossConfig, MSELossConfig, TrainingHyperParams
from regnn.model import ReGNNConfig, tree_routing_regularized_loss
from regnn.model.regnn import ReGNN
from regnn.macroutils import MacroConfig
from regnn.probe import ModeratedRegressionConfig


def test_tree_loss_config_creation():
    """Test TreeLossConfig creation with default values."""
    config = TreeLossConfig()
    assert config.name == "TreeLoss"
    assert config.lambda_tree == 0.01
    assert config.reduction == "mean"


def test_tree_loss_config_custom_lambda():
    """Test TreeLossConfig with custom lambda value."""
    config = TreeLossConfig(lambda_tree=0.05)
    assert config.lambda_tree == 0.05


def test_tree_loss_config_validation():
    """Test TreeLossConfig validation for non-negative lambda."""
    with pytest.raises(ValueError):
        TreeLossConfig(lambda_tree=-0.01)


def test_tree_routing_regularized_loss_function():
    """Test tree_routing_regularized_loss returns a callable."""
    loss_func = tree_routing_regularized_loss(lambda_tree=0.01, reduction="mean")
    assert callable(loss_func)


def test_tree_routing_loss_with_softtree():
    """Test tree routing loss with a SoftTree model."""
    # Create a simple ReGNN model with SoftTree
    config = ReGNNConfig.create(
        num_moderators=10,
        num_controlled=5,
        layer_input_sizes=[8],
        use_soft_tree=True,
        tree_depth=3,
        vae=False,
        batch_norm=False,
    )
    model = ReGNN.from_config(config)

    # Create loss function
    loss_func = tree_routing_regularized_loss(lambda_tree=0.01, reduction="mean")

    # Create sample data
    batch_size = 32
    moderators = torch.randn(batch_size, 1, 10)
    focal_predictor = torch.randn(batch_size, 1, 1)
    controlled_predictors = torch.randn(batch_size, 1, 5)

    # Forward pass
    predictions = model(moderators, focal_predictor, controlled_predictors)
    targets = torch.randn(batch_size, 1)

    # Compute loss
    loss = loss_func(predictions, targets, moderators, model)

    # Should return a scalar
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss > 0


def test_tree_routing_loss_without_softtree():
    """Test tree routing loss with non-SoftTree model (should only compute MSE)."""
    # Create a ReGNN model with MLP (not SoftTree)
    config = ReGNNConfig.create(
        num_moderators=10,
        num_controlled=5,
        layer_input_sizes=[8, 8],
        use_soft_tree=False,
        vae=False,
        batch_norm=False,
    )
    model = ReGNN.from_config(config)

    # Create loss function
    loss_func = tree_routing_regularized_loss(lambda_tree=0.01, reduction="mean")

    # Create sample data
    batch_size = 32
    moderators = torch.randn(batch_size, 1, 10)
    focal_predictor = torch.randn(batch_size, 1, 1)
    controlled_predictors = torch.randn(batch_size, 1, 5)

    # Forward pass
    predictions = model(moderators, focal_predictor, controlled_predictors)
    targets = torch.randn(batch_size, 1)

    # Compute loss
    loss = loss_func(predictions, targets, moderators, model)

    # Should still work (just MSE, no tree regularization)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss > 0


def test_tree_routing_loss_reduction_modes():
    """Test tree routing loss with different reduction modes."""
    config = ReGNNConfig.create(
        num_moderators=10,
        num_controlled=5,
        layer_input_sizes=[8],
        use_soft_tree=True,
        tree_depth=3,
        vae=False,
        batch_norm=False,
    )
    model = ReGNN.from_config(config)

    batch_size = 32
    moderators = torch.randn(batch_size, 1, 10)
    focal_predictor = torch.randn(batch_size, 1, 1)
    controlled_predictors = torch.randn(batch_size, 1, 5)

    predictions = model(moderators, focal_predictor, controlled_predictors)
    targets = torch.randn(batch_size, 1)

    # Test 'mean' reduction
    loss_mean = tree_routing_regularized_loss(lambda_tree=0.01, reduction="mean")
    loss_val_mean = loss_mean(predictions, targets, moderators, model)
    assert loss_val_mean.ndim == 0

    # Test 'sum' reduction
    loss_sum = tree_routing_regularized_loss(lambda_tree=0.01, reduction="sum")
    loss_val_sum = loss_sum(predictions, targets, moderators, model)
    assert loss_val_sum.ndim == 0
    assert loss_val_sum > loss_val_mean  # Sum should be larger than mean

    # Test 'none' reduction
    loss_none = tree_routing_regularized_loss(lambda_tree=0.01, reduction="none")
    loss_val_none = loss_none(predictions, targets, moderators, model)
    assert loss_val_none.shape == (batch_size, 1)


def test_tree_routing_loss_with_ensemble():
    """Test tree routing loss with SoftTree ensemble."""
    config = ReGNNConfig.create(
        num_moderators=10,
        num_controlled=5,
        layer_input_sizes=[8],
        use_soft_tree=True,
        tree_depth=3,
        n_ensemble=3,
        vae=False,
        batch_norm=False,
    )
    model = ReGNN.from_config(config)

    loss_func = tree_routing_regularized_loss(lambda_tree=0.01, reduction="mean")

    batch_size = 32
    moderators = torch.randn(batch_size, 1, 10)
    focal_predictor = torch.randn(batch_size, 1, 1)
    controlled_predictors = torch.randn(batch_size, 1, 5)

    predictions = model(moderators, focal_predictor, controlled_predictors)
    targets = torch.randn(batch_size, 1)

    loss = loss_func(predictions, targets, moderators, model)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss > 0


def test_tree_loss_config_with_softtree_model():
    """Test that TreeLossConfig works with SoftTree models."""
    # This validates the integration works at the config level
    config = ReGNNConfig.create(
        num_moderators=10,
        num_controlled=5,
        layer_input_sizes=[8],
        use_soft_tree=True,
        tree_depth=3,
        vae=False,
    )

    training_config = TrainingHyperParams(
        epochs=10, batch_size=32, loss_options=TreeLossConfig(lambda_tree=0.01)
    )

    # Should work without errors
    assert config.nn_config.use_soft_tree
    assert isinstance(training_config.loss_options, TreeLossConfig)


def test_tree_loss_config_validator_direct():
    """Test the validator logic directly."""
    from regnn.macroutils.base import MacroConfig

    # Test that the validator exists
    validators = [v for v in dir(MacroConfig) if "check_tree_loss" in v]
    assert len(validators) > 0
    assert "check_tree_loss_compatibility" in validators


def test_tree_loss_vs_mse_loss():
    """Test that TreeLossConfig produces different loss than MSELossConfig."""
    # Create same model for both
    config = ReGNNConfig.create(
        num_moderators=10,
        num_controlled=5,
        layer_input_sizes=[8],
        use_soft_tree=True,
        tree_depth=3,
        vae=False,
        batch_norm=False,
    )
    model1 = ReGNN.from_config(config)
    model2 = ReGNN.from_config(config)

    # Copy weights to ensure same model
    model2.load_state_dict(model1.state_dict())

    # Create data
    batch_size = 32
    moderators = torch.randn(batch_size, 1, 10)
    focal_predictor = torch.randn(batch_size, 1, 1)
    controlled_predictors = torch.randn(batch_size, 1, 5)
    targets = torch.randn(batch_size, 1)

    # Compute predictions
    with torch.no_grad():
        predictions1 = model1(moderators, focal_predictor, controlled_predictors)
        predictions2 = model2(moderators, focal_predictor, controlled_predictors)

    # TreeLossConfig with regularization
    tree_loss_func = tree_routing_regularized_loss(lambda_tree=0.01, reduction="mean")
    tree_loss = tree_loss_func(predictions1, targets, moderators, model1)

    # Pure MSE loss
    mse_loss = torch.mean((predictions2 - targets) ** 2)

    # TreeLoss should be higher due to regularization term
    assert tree_loss > mse_loss
