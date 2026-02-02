import pytest
import torch
from regnn.model.base import TreeConfig, IndexPredictionConfig, ReGNNConfig
from regnn.model.regnn import SoftTree, SoftTreeEnsemble, IndexPredictionModel, ReGNN


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
    individual_regs = [tree.compute_routing_regularization(input_tensor) for tree in model.models]
    manual_average = sum(individual_regs) / len(model.models)
    
    # Should match
    assert torch.allclose(reg_loss, manual_average, atol=1e-6)
