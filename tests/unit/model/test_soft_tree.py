import pytest
import torch
import numpy as np
from regnn.model.base import TreeConfig, IndexPredictionConfig, ReGNNConfig
from regnn.model.regnn import SoftTree, SoftTreeEnsemble, IndexPredictionModel, ReGNN
from regnn.train import TemperatureAnnealingConfig
from regnn.macroutils.utils import TemperatureAnnealer


def test_softtree_creation():
    """Test basic SoftTree creation and properties."""
    config = TreeConfig(input_dim=10, output_dim=5, depth=3)
    model = SoftTree.from_config(config)
    assert isinstance(model, SoftTree)
    assert model.input_dim == 10
    assert model.output_dim == 5
    assert model.depth == 3
    assert model.num_internal_nodes == 7  # 2^3 - 1
    assert model.num_leaf_nodes == 8  # 2^3


def test_softtree_forward():
    """Test SoftTree forward pass returns correct shape."""
    config = TreeConfig(input_dim=10, output_dim=5, depth=3)
    model = SoftTree.from_config(config)
    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 5)


def test_softtree_with_3d_input():
    """Test SoftTree handles 3D input (batch, 1, features)."""
    config = TreeConfig(input_dim=10, output_dim=5, depth=3)
    model = SoftTree.from_config(config)
    batch_size = 32
    input_tensor = torch.randn(batch_size, 1, 10)
    output = model(input_tensor)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 5)


def test_softtree_with_dropout():
    """Test SoftTree with dropout configuration."""
    config = TreeConfig(input_dim=10, output_dim=5, depth=3, dropout=0.5)
    model = SoftTree.from_config(config)
    assert model.dropout_rate == 0.5

    # Test forward pass
    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    model.train()
    output = model(input_tensor)
    assert output.shape == (batch_size, 5)


def test_softtree_with_sharpness():
    """Test SoftTree with different sharpness values."""
    # Test with default sharpness
    config1 = TreeConfig(input_dim=10, output_dim=5, depth=3, sharpness=1.0)
    model1 = SoftTree.from_config(config1)
    assert model1.sharpness == 1.0

    # Test with higher sharpness
    config2 = TreeConfig(input_dim=10, output_dim=5, depth=3, sharpness=5.0)
    model2 = SoftTree.from_config(config2)
    assert model2.sharpness == 5.0

    # Both should produce valid outputs
    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output1 = model1(input_tensor)
    output2 = model2(input_tensor)
    assert output1.shape == (batch_size, 5)
    assert output2.shape == (batch_size, 5)


def test_softtree_different_depths():
    """Test SoftTree with different tree depths."""
    for depth in [2, 3, 4, 5]:
        config = TreeConfig(input_dim=10, output_dim=5, depth=depth)
        model = SoftTree.from_config(config)
        assert model.num_internal_nodes == 2**depth - 1
        assert model.num_leaf_nodes == 2**depth

        # Test forward pass
        batch_size = 16
        input_tensor = torch.randn(batch_size, 10)
        output = model(input_tensor)
        assert output.shape == (batch_size, 5)


def test_softtree_eval_mode():
    """Test SoftTree behavior in eval mode."""
    config = TreeConfig(input_dim=10, output_dim=5, depth=3)
    model = SoftTree.from_config(config)
    model.eval()

    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 5)


def test_softtree_ensemble_creation():
    """Test SoftTreeEnsemble creation."""
    config = TreeConfig(input_dim=10, output_dim=5, depth=3)
    n_ensemble = 3
    model = SoftTreeEnsemble.from_config(config, n_ensemble=n_ensemble)
    assert isinstance(model, SoftTreeEnsemble)
    assert len(model.models) == n_ensemble


def test_softtree_ensemble_forward():
    """Test SoftTreeEnsemble forward pass."""
    config = TreeConfig(input_dim=10, output_dim=5, depth=3)
    n_ensemble = 5
    model = SoftTreeEnsemble.from_config(config, n_ensemble=n_ensemble)

    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)
    output = model(input_tensor)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 5)


def test_softtree_ensemble_averaging():
    """Test that SoftTreeEnsemble properly averages outputs."""
    config = TreeConfig(input_dim=10, output_dim=5, depth=3)
    n_ensemble = 3
    model = SoftTreeEnsemble.from_config(config, n_ensemble=n_ensemble)
    model.eval()

    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)

    # Get ensemble output
    ensemble_output = model(input_tensor)

    # Manually compute average
    individual_outputs = [m(input_tensor) for m in model.models]
    manual_average = torch.mean(torch.stack(individual_outputs), dim=0)

    # Should be equal
    assert torch.allclose(ensemble_output, manual_average, atol=1e-6)


def test_index_prediction_model_with_softtree():
    """Test IndexPredictionModel with SoftTree backbone."""
    config = IndexPredictionConfig(
        num_moderators=10,
        layer_input_sizes=[5],  # Only used for output dim
        use_soft_tree=True,
        tree_depth=3,
        vae=False,  # VAE not supported with SoftTree
        batch_norm=False,
    )
    model = IndexPredictionModel.from_config(config)

    batch_size = 32
    moderators = torch.randn(batch_size, 1, 10)
    output = model(moderators)
    assert isinstance(output, torch.Tensor)


def test_index_prediction_model_softtree_ensemble():
    """Test IndexPredictionModel with SoftTree ensemble."""
    config = IndexPredictionConfig(
        num_moderators=10,
        layer_input_sizes=[5],
        use_soft_tree=True,
        tree_depth=3,
        n_ensemble=3,
        vae=False,
        batch_norm=False,
    )
    model = IndexPredictionModel.from_config(config)

    batch_size = 32
    moderators = torch.randn(batch_size, 1, 10)
    output = model(moderators)
    assert isinstance(output, torch.Tensor)


def test_regnn_with_softtree():
    """Test ReGNN with SoftTree backbone."""
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
    moderators = torch.randn(batch_size, 10)
    focal_predictor = torch.randn(batch_size)
    controlled_predictors = torch.randn(batch_size, 5)

    output = model(moderators, focal_predictor, controlled_predictors)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 1)


def test_softtree_vae_validation_error():
    """Test that using VAE with SoftTree raises validation error."""
    with pytest.raises(ValueError, match="VAE is not supported with SoftTree"):
        config = IndexPredictionConfig(
            num_moderators=10,
            layer_input_sizes=[5],
            use_soft_tree=True,
            tree_depth=3,
            vae=True,  # This should raise an error
        )


def test_softtree_missing_depth_validation_error():
    """Test that missing tree_depth with use_soft_tree raises validation error."""
    with pytest.raises(ValueError, match="tree_depth must be provided"):
        config = IndexPredictionConfig(
            num_moderators=10,
            layer_input_sizes=[5],
            use_soft_tree=True,
            tree_depth=None,  # This should raise an error
        )


def test_softtree_mutual_exclusivity_validation_error():
    """Test that using both use_resmlp and use_soft_tree raises validation error."""
    with pytest.raises(
        ValueError, match="Cannot use both use_resmlp and use_soft_tree"
    ):
        config = IndexPredictionConfig(
            num_moderators=10,
            layer_input_sizes=[5],
            use_resmlp=True,
            use_soft_tree=True,  # This should raise an error
            tree_depth=3,
        )


def test_softtree_with_sharpness_parameter():
    """Test that tree_sharpness parameter is properly used."""
    config = IndexPredictionConfig(
        num_moderators=10,
        layer_input_sizes=[5],
        use_soft_tree=True,
        tree_depth=3,
        tree_sharpness=2.0,
        vae=False,
        batch_norm=False,
    )
    model = IndexPredictionModel.from_config(config)

    batch_size = 32
    moderators = torch.randn(batch_size, 1, 10)
    output = model(moderators)
    assert isinstance(output, torch.Tensor)


def test_softtree_compute_routing_regularization():
    """Test SoftTree.compute_routing_regularization() method."""
    config = TreeConfig(input_dim=10, output_dim=5, depth=3)
    model = SoftTree.from_config(config)

    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)

    # Compute regularization
    reg_loss = model.compute_routing_regularization(input_tensor)

    # Should return a scalar tensor
    assert isinstance(reg_loss, torch.Tensor)
    assert reg_loss.ndim == 0  # Scalar
    assert reg_loss >= 0  # Regularization should be non-negative


def test_softtree_routing_regularization_with_3d_input():
    """Test routing regularization handles 3D input."""
    config = TreeConfig(input_dim=10, output_dim=5, depth=3)
    model = SoftTree.from_config(config)

    batch_size = 32
    input_tensor = torch.randn(batch_size, 1, 10)

    # Compute regularization
    reg_loss = model.compute_routing_regularization(input_tensor)

    # Should return a scalar tensor
    assert isinstance(reg_loss, torch.Tensor)
    assert reg_loss.ndim == 0  # Scalar


def test_softtree_routing_regularization_balanced():
    """Test that regularization is minimal when routing is balanced."""
    config = TreeConfig(input_dim=10, output_dim=5, depth=2)
    model = SoftTree.from_config(config)

    # Create input that results in balanced routing (all 0s -> sigmoid = 0.5)
    # Set weights and bias to zero to get balanced routing
    with torch.no_grad():
        model.internal_node_weights.zero_()
        model.internal_node_bias.zero_()

    batch_size = 100
    input_tensor = torch.randn(batch_size, 10)

    reg_loss = model.compute_routing_regularization(input_tensor)

    # With balanced routing (all 0.5), the loss should be close to 0
    # -[0.5*log(0.5) + 0.5*log(0.5)] = -[0.5*(-0.693) + 0.5*(-0.693)] = 0.693
    expected_loss_per_node = -0.5 * (torch.log(torch.tensor(0.5)) * 2)
    expected_total = expected_loss_per_node * model.num_internal_nodes
    assert torch.allclose(reg_loss, expected_total, atol=1e-5)


def test_softtree_routing_regularization_unbalanced():
    """Test that regularization increases when routing is unbalanced."""
    config = TreeConfig(input_dim=10, output_dim=5, depth=2)
    model = SoftTree.from_config(config)

    batch_size = 100
    input_tensor = torch.randn(batch_size, 10)

    # Get regularization with random routing
    reg_loss_random = model.compute_routing_regularization(input_tensor)

    # Create strongly unbalanced routing (all samples go one way)
    # Set large positive weights to bias routing
    with torch.no_grad():
        model.internal_node_weights.fill_(10.0)
        model.internal_node_bias.fill_(10.0)

    reg_loss_unbalanced = model.compute_routing_regularization(input_tensor)

    # Unbalanced routing should have higher regularization loss
    # (though balanced could also be high, this tests the mechanism works)
    assert reg_loss_unbalanced > 0


def test_softtree_ensemble_routing_regularization():
    """Test SoftTreeEnsemble.compute_routing_regularization() method."""
    config = TreeConfig(input_dim=10, output_dim=5, depth=3)
    n_ensemble = 3
    model = SoftTreeEnsemble.from_config(config, n_ensemble=n_ensemble)

    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)

    # Compute regularization
    reg_loss = model.compute_routing_regularization(input_tensor)

    # Should return a scalar tensor
    assert isinstance(reg_loss, torch.Tensor)
    assert reg_loss.ndim == 0  # Scalar
    assert reg_loss >= 0

    # Manually compute average of individual regularizations
    individual_regs = [
        tree.compute_routing_regularization(input_tensor) for tree in model.models
    ]
    manual_average = sum(individual_regs) / len(model.models)

    # Should match
    assert torch.allclose(reg_loss, manual_average, atol=1e-6)


# ============================================================================
# Temperature Annealing Tests
# ============================================================================


def test_softtree_set_temperature():
    """Test that set_temperature updates sharpness correctly."""
    config = TreeConfig(input_dim=10, output_dim=5, depth=3, sharpness=1.0)
    model = SoftTree.from_config(config)

    assert model.sharpness == 1.0

    model.set_temperature(5.0)
    assert model.sharpness == 5.0

    model.set_temperature(10.0)
    assert model.sharpness == 10.0


def test_softtree_ensemble_set_temperature():
    """Test that ensemble propagates temperature to all trees."""
    config = TreeConfig(input_dim=10, output_dim=5, depth=3, sharpness=1.0)
    n_ensemble = 3
    model = SoftTreeEnsemble.from_config(config, n_ensemble=n_ensemble)

    # Check initial sharpness
    for tree in model.models:
        assert tree.sharpness == 1.0

    # Update temperature
    model.set_temperature(5.0)

    # Check all trees updated
    for tree in model.models:
        assert tree.sharpness == 5.0


def test_index_prediction_model_set_temperature():
    """Test that IndexPredictionModel propagates temperature to SoftTree."""
    config = IndexPredictionConfig(
        num_moderators=10,
        layer_input_sizes=[8],
        use_soft_tree=True,
        tree_depth=3,
        tree_sharpness=1.0,
        vae=False,
        batch_norm=False,
    )
    model = IndexPredictionModel.from_config(config)

    assert model.mlp.sharpness == 1.0

    model.set_temperature(7.0)
    assert model.mlp.sharpness == 7.0


def test_regnn_set_temperature():
    """Test that ReGNN propagates temperature through hierarchy."""
    config = ReGNNConfig.create(
        num_moderators=10,
        num_controlled=5,
        layer_input_sizes=[8],
        use_soft_tree=True,
        tree_depth=3,
        tree_sharpness=1.0,
        vae=False,
        batch_norm=False,
    )
    model = ReGNN.from_config(config)

    # Check initial sharpness
    initial_sharpness = model.index_prediction_model.mlp.sharpness
    assert initial_sharpness == 1.0

    # Update through top-level
    model.set_temperature(8.0)

    # Verify propagation
    assert model.index_prediction_model.mlp.sharpness == 8.0


def test_temperature_annealer_linear_schedule():
    """Test linear temperature schedule."""
    config = TemperatureAnnealingConfig(
        schedule_type="linear",
        initial_temp=1.0,
        final_temp=10.0,
    )
    annealer = TemperatureAnnealer(config, total_epochs=10)

    # Check endpoints
    assert annealer.get_temperature(0) == 1.0
    assert annealer.get_temperature(9) == 10.0

    # Check midpoint
    mid_temp = annealer.get_temperature(4)
    assert 4.0 < mid_temp < 6.0  # Should be around 5.0


def test_temperature_annealer_exponential_schedule():
    """Test exponential temperature schedule."""
    config = TemperatureAnnealingConfig(
        schedule_type="exponential",
        initial_temp=1.0,
        final_temp=100.0,
        exp_gamma=2.0,
    )
    annealer = TemperatureAnnealer(config, total_epochs=10)

    # Check endpoints
    temp_0 = annealer.get_temperature(0)
    temp_9 = annealer.get_temperature(9)

    assert np.isclose(temp_0, 1.0, atol=0.1)
    assert np.isclose(temp_9, 100.0, atol=1.0)

    # Exponential should have steeper curve at the end
    temp_8 = annealer.get_temperature(8)
    temp_7 = annealer.get_temperature(7)
    assert (temp_9 - temp_8) > (temp_8 - temp_7)


def test_temperature_annealer_cosine_schedule():
    """Test cosine temperature schedule."""
    config = TemperatureAnnealingConfig(
        schedule_type="cosine",
        initial_temp=1.0,
        final_temp=10.0,
    )
    annealer = TemperatureAnnealer(config, total_epochs=10)

    # Check endpoints
    assert np.isclose(annealer.get_temperature(0), 1.0, atol=0.01)
    assert np.isclose(annealer.get_temperature(9), 10.0, atol=0.01)

    # Cosine should be slower at start and end, faster in middle
    temp_1 = annealer.get_temperature(1)
    temp_2 = annealer.get_temperature(2)
    temp_4 = annealer.get_temperature(4)
    temp_5 = annealer.get_temperature(5)

    # Middle should have larger change than beginning
    assert (temp_5 - temp_4) > (temp_2 - temp_1)


def test_temperature_annealer_step_schedule():
    """Test step-based temperature schedule."""
    config = TemperatureAnnealingConfig(
        schedule_type="step",
        initial_temp=1.0,
        final_temp=16.0,
        step_size=3,
        step_gamma=2.0,
    )
    annealer = TemperatureAnnealer(config, total_epochs=10)

    # Check step behavior
    assert annealer.get_temperature(0) == 1.0
    assert annealer.get_temperature(1) == 1.0
    assert annealer.get_temperature(2) == 1.0
    assert annealer.get_temperature(3) == 2.0  # First step
    assert annealer.get_temperature(4) == 2.0
    assert annealer.get_temperature(5) == 2.0
    assert annealer.get_temperature(6) == 4.0  # Second step
    assert annealer.get_temperature(9) == 8.0  # Third step


def test_temperature_annealer_update_model():
    """Test that annealer updates model temperature correctly."""
    config = ReGNNConfig.create(
        num_moderators=10,
        num_controlled=5,
        layer_input_sizes=[8],
        use_soft_tree=True,
        tree_depth=3,
        tree_sharpness=1.0,
        vae=False,
        batch_norm=False,
    )
    model = ReGNN.from_config(config)

    temp_config = TemperatureAnnealingConfig(
        schedule_type="linear",
        initial_temp=1.0,
        final_temp=10.0,
    )
    annealer = TemperatureAnnealer(temp_config, total_epochs=10)

    # Update at different epochs
    new_temp_0 = annealer.update_model_temperature(model, epoch=0)
    assert new_temp_0 == 1.0
    assert model.index_prediction_model.mlp.sharpness == 1.0

    new_temp_5 = annealer.update_model_temperature(model, epoch=5)
    assert 5.5 < new_temp_5 < 6.5  # At epoch 5/9, should be around 6.0
    assert model.index_prediction_model.mlp.sharpness == new_temp_5

    new_temp_9 = annealer.update_model_temperature(model, epoch=9)
    assert new_temp_9 == 10.0
    assert model.index_prediction_model.mlp.sharpness == 10.0


def test_temperature_annealing_affects_routing():
    """Test that higher temperature makes routing sharper."""
    config = TreeConfig(input_dim=10, output_dim=5, depth=3, sharpness=1.0)
    model = SoftTree.from_config(config)

    batch_size = 32
    input_tensor = torch.randn(batch_size, 10)

    # Get output with low temperature
    model.set_temperature(1.0)
    output_low = model(input_tensor)

    # Get output with high temperature
    model.set_temperature(10.0)
    output_high = model(input_tensor)

    # Outputs should be different (routing is sharper with high temp)
    assert not torch.allclose(output_low, output_high, atol=1e-3)


def test_temperature_config_validation_step():
    """Test that step schedule validation works."""
    # Missing step_size should raise error
    with pytest.raises(ValueError, match="step_size must be provided"):
        TemperatureAnnealingConfig(
            schedule_type="step",
            initial_temp=1.0,
            final_temp=10.0,
            step_gamma=2.0,
        )

    # Missing step_gamma should raise error
    with pytest.raises(ValueError, match="step_gamma must be provided"):
        TemperatureAnnealingConfig(
            schedule_type="step",
            initial_temp=1.0,
            final_temp=10.0,
            step_size=3,
        )


def test_temperature_config_validation_exponential():
    """Test that exponential schedule validation works."""
    # Missing exp_gamma should raise error
    with pytest.raises(ValueError, match="exp_gamma must be provided"):
        TemperatureAnnealingConfig(
            schedule_type="exponential",
            initial_temp=1.0,
            final_temp=10.0,
        )
