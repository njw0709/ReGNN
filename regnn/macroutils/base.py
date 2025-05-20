from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from regnn.eval.base import EvaluationOptions
from regnn.model.base import ReGNNConfig
from regnn.train.base import TrainingHyperParams, ProbeOptions


class MacroConfig(BaseModel):
    """Configuration for ReGNN training"""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    # Core configurations
    model: ReGNNConfig = Field(..., description="Model configuration")
    training: TrainingHyperParams = Field(
        default_factory=TrainingHyperParams, description="Training hyperparameters"
    )
    evaluation: EvaluationOptions = Field(..., description="Evaluation configuration")
    # early_stopping: EarlyStoppingConfig = Field(
    #     default_factory=EarlyStoppingConfig, description="Early stopping configuration"
    # )
    probe: ProbeOptions = Field(
        default_factory=ProbeOptions,
        description="While-training probe configuration",
    )
