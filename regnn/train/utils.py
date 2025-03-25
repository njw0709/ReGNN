import torch
from regnn.model.regnn import ReGNN
from typing import Dict, List, Optional
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


def save_regnn(
    model: ReGNN,
    save_dir: str = os.path.join(TEMP_DIR, "checkpoints"),
    data_id: Optional[str] = None,
) -> None:
    """Save ReGNN model to disk"""
    if data_id is not None:
        model_name = os.path.join(save_dir, f"regnn_model_{data_id}.pt")
    else:
        num_files = len([f for f in os.listdir(save_dir) if f.endswith(".pt")])
        model_name = os.path.join(save_dir, f"regnn_model_{num_files}.pt")
    torch.save(model.state_dict(), model_name)
