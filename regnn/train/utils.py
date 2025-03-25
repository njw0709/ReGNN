import torch
from regnn.model.regnn import ReGNN
from typing import Dict, List, Optional, Union
import os
from regnn.train.constants import TEMP_DIR


def get_gradient_norms(model: ReGNN) -> Dict[str, List[float]]:
    """Calculate gradient norms for model parameters"""
    grad_norms = {}
    main_parameters = [model.focal_predictor_main_weight, model.predicted_index_weight]
    main_parameters += [p for p in model.controlled_var_weights.parameters()]
    grad_main = [p.grad.norm(2).item() for p in main_parameters]
    index_model_params = [p for p in model.index_prediction_model.parameters()]
    grad_index = [p.grad.norm(2).item() for p in index_model_params]
    grad_norms["main"] = grad_main
    grad_norms["index"] = grad_index
    return grad_norms


def get_l2_length(model: ReGNN) -> Dict[str, float]:
    """Calculate L2 norms for model parameters"""
    l2_lengths = {}
    main_parameters = [model.focal_predictor_main_weight, model.predicted_index_weight]
    main_parameters += [p for p in model.controlled_var_weights.parameters()][:-1]
    main_parameters = torch.cat(main_parameters, dim=1)
    main_param_l2 = main_parameters.norm(2).item()

    index_norm = model.predicted_index_weight.norm(2).item()
    l2_lengths["main"] = main_param_l2
    l2_lengths["index"] = index_norm
    return l2_lengths


def save_model(
    model: torch.nn.Module,
    model_type: str = "model",
    save_dir: str = os.path.join(TEMP_DIR, "checkpoints"),
    data_id: Optional[str] = None,
) -> str:
    """Save PyTorch model to disk

    Args:
        model: PyTorch model to save
        model_type: Type of model for filename prefix (e.g. 'regnn', 'mlp')
        save_dir: Directory to save model in
        data_id: Optional identifier to include in filename

    Returns:
        str: Path to saved model file
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Generate filename
    if data_id is not None:
        model_name = os.path.join(save_dir, f"{model_type}_{data_id}.pt")
    else:
        num_files = len([f for f in os.listdir(save_dir) if f.endswith(".pt")])
        model_name = os.path.join(save_dir, f"{model_type}_{num_files}.pt")

    # Save model
    torch.save(model.state_dict(), model_name)
    return model_name


def save_regnn(
    model: ReGNN,
    save_dir: str = os.path.join(TEMP_DIR, "checkpoints"),
    data_id: Optional[str] = None,
) -> None:
    """Save ReGNN model to disk"""
    save_model(model, model_type="regnn", save_dir=save_dir, data_id=data_id)


def load_model(
    model: torch.nn.Module,
    model_path: str,
    map_location: Optional[Union[str, torch.device]] = None,
) -> torch.nn.Module:
    """Load PyTorch model from disk

    Args:
        model: Instantiated PyTorch model to load weights into
        model_path: Path to saved model file
        map_location: Optional device to map model to (e.g. 'cpu', 'cuda')

    Returns:
        torch.nn.Module: Model with loaded weights
    """
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    return model
