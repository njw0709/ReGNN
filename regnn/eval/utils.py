import stata_setup
import shap
import torch
from regnn.model.regnn import ReGNN
import numpy as np
import pandas as pd
from regnn.constants import TEMP_DIR
import os
from typing import Union

# Track Stata initialization status
_stata_initialized = False


def init_stata():
    global _stata_initialized
    if not _stata_initialized:
        stata_setup.config("/usr/local/stata17", "mp")
        _stata_initialized = True
    from pystata import stata

    return stata


def init_shap():
    shap.initjs()


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
