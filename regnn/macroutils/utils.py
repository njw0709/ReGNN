import torch
from regnn.model.regnn import ReGNN
from typing import Dict, List, Optional, Union
import os
from regnn.constants import TEMP_DIR
import pandas as pd
import numpy as np


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


def compute_index_prediction(
    model: ReGNN, interaction_predictors: torch.Tensor
) -> np.ndarray:

    index_model = model.index_prediction_model
    index_model.to(interaction_predictors.device).eval()
    if index_model.vae:
        index_prediction, log_var = index_model(interaction_predictors)
    else:
        index_prediction = index_model(interaction_predictors)
    index_prediction = index_prediction.detach().cpu().numpy()

    return index_prediction


def save_intermediate_df(
    df_orig: pd.DataFrame,
    index_predictions: np.ndarray,
    output_index_name: str,
    save_dir: str = os.path.join(TEMP_DIR, "data"),
    data_id: Union[str, None] = None,
    interaction_direction: str = "positive",
):
    if data_id is not None:
        save_path = os.path.join(save_dir, f"index_prediction_{data_id}.dta")
    else:
        num_files = len(os.listdir(save_dir))
        save_path = os.path.join(save_dir, f"index_prediction_{num_files}.dta")
    # save to file
    if interaction_direction == "positive":
        output_index_name = "res_index"
    elif interaction_direction == "negative":
        output_index_name = "vul_index"
    else:
        raise ValueError("interaction Direction must either be positive or negative!!")

    df_orig[output_index_name] = index_predictions
    df_orig.to_stata(save_path, write_index=False)
    return True
