import pytest
import torch
import numpy as np
from regnn.model.base import IndexPredictionConfig, SVDConfig
from regnn.model.regnn import IndexPredictionModel


def test_index_prediction_model_creation_basic_no_svd_no_vae():
    config = IndexPredictionConfig(
        num_moderators=10,
        layer_input_sizes=[
            5,
        ],
        vae=False,
    )  # Removed svd parameter to use default factory
    model = IndexPredictionModel.from_config(config)
    assert isinstance(model, IndexPredictionModel)
    batch_size = 2
    moderators = torch.randn(batch_size, 10)
    output = model(moderators)
    assert output.shape == (batch_size, 1)


def test_index_prediction_model_creation_multiple_no_svd_no_vae():
    config = IndexPredictionConfig(
        num_moderators=[5, 5], layer_input_sizes=[[3, 2], [3, 2]], vae=False
    )
    model = IndexPredictionModel.from_config(config)
    assert isinstance(model, IndexPredictionModel)
    batch_size = 2
    moderators1 = torch.randn(batch_size, 5)
    moderators2 = torch.randn(batch_size, 5)
    moderators = [moderators1, moderators2]
    output = model(moderators)
    assert isinstance(output, list)
    for o in output:
        assert o.shape == (batch_size, 1)


def test_index_prediction_model_creation_basic_with_svd_no_vae():
    num_moderators = 10
    k_dim = 5
    svd_matrix = np.random.rand(num_moderators, num_moderators).astype(np.float32)
    config = IndexPredictionConfig(
        num_moderators=num_moderators,
        layer_input_sizes=[5, 2],
        svd=SVDConfig(enabled=True, k_dim=k_dim, svd_matrix=svd_matrix),
        vae=False,
    )
    model = IndexPredictionModel.from_config(config)
    assert isinstance(model, IndexPredictionModel)
    batch_size = 2
    moderators = torch.randn(batch_size, num_moderators)
    output = model(moderators)
    assert output.shape == (batch_size, 1)


def test_index_prediction_model_creation_multiple_with_svd_no_vae():
    num_moderators = [5, 5]
    k_dim = [2, 2]
    svd_matrix = [
        np.random.rand(num_moderators[i], num_moderators[i]).astype(np.float32)
        for i in range(len(num_moderators))
    ]
    config = IndexPredictionConfig(
        num_moderators=num_moderators,
        layer_input_sizes=[[3, 2], [3, 2]],
        vae=False,
        svd=SVDConfig(enabled=True, k_dim=k_dim, svd_matrix=svd_matrix),
    )
    model = IndexPredictionModel.from_config(config)
    assert isinstance(model, IndexPredictionModel)
    batch_size = 2
    moderators1 = torch.randn(batch_size, 5)
    moderators2 = torch.randn(batch_size, 5)
    moderators = [moderators1, moderators2]
    output = model(moderators)
    assert isinstance(output, list)
    for o in output:
        assert o.shape == (batch_size, 1)


def test_index_prediction_model_creation_basic_no_svd_with_vae():
    config = IndexPredictionConfig(
        num_moderators=10, layer_input_sizes=[5, 2], vae=True
    )  # Removed svd parameter to use default factory
    model = IndexPredictionModel.from_config(config)
    assert isinstance(model, IndexPredictionModel)
    batch_size = 2
    moderators = torch.randn(batch_size, 10)
    output = model(moderators)
    assert output.shape == (batch_size, 1)


def test_index_prediction_model_creation_basic_no_svd_with_vae_output_mu_var():
    config = IndexPredictionConfig(
        num_moderators=10,
        layer_input_sizes=[5, 2],
        vae=True,
        output_mu_var=True,
    )  # Removed svd parameter to use default factory
    model = IndexPredictionModel.from_config(config)
    assert isinstance(model, IndexPredictionModel)
    batch_size = 2
    moderators = torch.randn(batch_size, 10)
    output = model(moderators)
    assert isinstance(output, tuple)
    assert len(output) == 3
    assert output[0].shape == (batch_size, 1)
    assert output[1].shape == (batch_size, 1)
    assert output[2].shape == (batch_size, 1)


def test_index_prediction_model_creation_multiple_no_svd_with_vae_output_mu_var():
    config = IndexPredictionConfig(
        num_moderators=[5, 5],
        layer_input_sizes=[[3, 2], [3, 2]],
        vae=True,
        output_mu_var=True,
    )  # Removed svd parameter to use default factory
    model = IndexPredictionModel.from_config(config)
    assert isinstance(model, IndexPredictionModel)
    batch_size = 2
    moderators1 = torch.randn(batch_size, 5)
    moderators2 = torch.randn(batch_size, 5)
    moderators = [moderators1, moderators2]
    output = model(moderators)
    assert isinstance(output, tuple)
    assert len(output) == 3
    assert isinstance(output[0], list)
    for o in output[0]:
        assert o.shape == (batch_size, 1)
    assert isinstance(output[1], list)
    for o in output[1]:
        assert o.shape == (batch_size, 1)
    assert isinstance(output[2], list)
    for o in output[2]:
        assert o.shape == (batch_size, 1)


def test_index_prediction_model_creation_basic_with_svd_with_vae_output_mu_var():
    num_moderators = 10
    k_dim = 5
    svd_matrix = np.random.rand(num_moderators, num_moderators).astype(np.float32)
    config = IndexPredictionConfig(
        num_moderators=num_moderators,
        layer_input_sizes=[5, 2],
        svd=SVDConfig(enabled=True, k_dim=k_dim, svd_matrix=svd_matrix),
        vae=True,
        output_mu_var=True,
    )
    model = IndexPredictionModel.from_config(config)
    assert isinstance(model, IndexPredictionModel)
    batch_size = 2
    moderators = torch.randn(batch_size, num_moderators)
    output = model(moderators)
    assert isinstance(output, tuple)
    assert len(output) == 3
    assert output[0].shape == (batch_size, 1)
    assert output[1].shape == (batch_size, 1)
    assert output[2].shape == (batch_size, 1)


def test_index_prediction_model_creation_multiple_with_svd_with_vae_output_mu_var():
    num_moderators = [5, 5]
    k_dim = [2, 2]
    svd_matrix = [
        np.random.rand(num_moderators[i], num_moderators[i]).astype(np.float32)
        for i in range(len(num_moderators))
    ]
    config = IndexPredictionConfig(
        num_moderators=num_moderators,
        layer_input_sizes=[[3, 2], [3, 2]],
        svd=SVDConfig(enabled=True, k_dim=k_dim, svd_matrix=svd_matrix),
        vae=True,
        output_mu_var=True,
    )
    model = IndexPredictionModel.from_config(config)
    assert isinstance(model, IndexPredictionModel)
    batch_size = 2
    moderators1 = torch.randn(batch_size, 5)
    moderators2 = torch.randn(batch_size, 5)
    moderators = [moderators1, moderators2]
    output = model(moderators)
    assert isinstance(output, tuple)
    assert len(output) == 3
    assert isinstance(output[0], list)
    for o in output[0]:
        assert o.shape == (batch_size, 1)
    assert isinstance(output[1], list)
    for o in output[1]:
        assert o.shape == (batch_size, 1)
    assert isinstance(output[2], list)
    for o in output[2]:
        assert o.shape == (batch_size, 1)
