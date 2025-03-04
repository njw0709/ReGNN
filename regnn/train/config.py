from typing import Sequence, Union, Optional, List
from pydantic import BaseModel, Field
import torch


class TrainingConfig(BaseModel):
    """Configuration for ReGNN training"""

    hidden_layer_sizes: List[int]
    vae: bool = False
    svd: bool = False
    k_dims: int = 10
    epochs: int = 100
    batch_size: int = 32
    lr: float = 0.001
    weight_decay_regression: float = 0.0
    weight_decay_nn: float = 0.0
    regress_cmd: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    shuffle: bool = True
    evaluate: bool = False
    eval_epoch: int = 10
    get_testset_results: bool = True
    file_id: Optional[str] = None
    save_model: bool = False
    use_stata: bool = False
    return_trajectory: bool = False
    vae_loss: bool = False
    vae_lambda: float = 0.1
    dropout: float = 0.1
    n_models: int = 1
    elasticnet: bool = False
    lasso: bool = False
    lambda_reg: float = 0.1
    survey_weights: bool = True
    include_bias_focal_predictor: bool = True
    interaction_direction: str = "positive"
    get_l2_lengths: bool = True
    early_stop: bool = True
    early_stop_criterion: float = 0.01
    stop_after: int = 100
    save_intermediate_index: bool = False

    @property
    def validate_hidden_layer_sizes(self):
        if not self.hidden_layer_sizes:
            raise ValueError("hidden_layer_sizes cannot be empty")

        # Check if it's a list of lists
        if isinstance(self.hidden_layer_sizes[0], list):
            for layer_sizes in self.hidden_layer_sizes:
                if not layer_sizes:
                    raise ValueError("Layer sizes cannot be empty")
                if layer_sizes[-1] != 1:
                    raise ValueError("Last layer size must be 1 for all models")
        else:
            # Single model case
            if self.hidden_layer_sizes[-1] != 1:
                raise ValueError("Last layer size must be 1")


class TrajectoryData(BaseModel):
    """Data structure for tracking training trajectory"""

    train_loss: float = -1
    test_loss: float = -1
    regression_summary: Optional[dict] = None
    regression_summary_test: Optional[dict] = None
    l2: Optional[List[dict]] = None

    class Config:
        arbitrary_types_allowed = True
