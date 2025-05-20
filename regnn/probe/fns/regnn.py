from typing import Dict, List
import torch
from regnn.model.regnn import ReGNN


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
