import torch
from regnn.model.regnn import ReGNN
from typing import Optional, Union, Tuple, List, Dict, Any
import numpy as np
import torch.nn as nn
import torch.optim as optim
from regnn.model import (
    vae_kld_regularized_loss,
    tree_routing_regularized_loss,
    prior_penalty_loss,
    elasticnet_loss,
    lasso_loss,
)
from regnn.train import (
    TrainingHyperParams,
    MSELossConfig,
    KLDLossConfig,
    TreeLossConfig,
    PriorPenaltyLossConfig,
    ElasticNetRegConfig,
    StepLRConfig,
    ExponentialLRConfig,
    CosineAnnealingLRConfig,
    ReduceLROnPlateauConfig,
    WarmupCosineConfig,
    SchedulerConfigUnion,
    TemperatureAnnealingConfig,
)


# Imports for the new function
from regnn.probe import (
    ProbeData,
    ObjectiveProbe,
    OLSModeratedResultsProbe,
    L2NormProbe,
    DataSource,
)


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


def create_lr_scheduler(
    optimizer: optim.Optimizer,
    scheduler_config: Optional[SchedulerConfigUnion],
    total_epochs: int,
) -> Optional[Union[
    optim.lr_scheduler.StepLR,
    optim.lr_scheduler.ExponentialLR,
    optim.lr_scheduler.CosineAnnealingLR,
    optim.lr_scheduler.ReduceLROnPlateau,
    optim.lr_scheduler.SequentialLR,
]]:
    """Create learning rate scheduler from configuration.

    Args:
        optimizer: Optimizer to schedule
        scheduler_config: Scheduler configuration
        total_epochs: Total number of training epochs

    Returns:
        Scheduler instance or None if no scheduler configured
    """
    if scheduler_config is None:
        return None

    if isinstance(scheduler_config, StepLRConfig):
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.step_size,
            gamma=scheduler_config.gamma,
        )
    elif isinstance(scheduler_config, ExponentialLRConfig):
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_config.gamma,
        )
    elif isinstance(scheduler_config, CosineAnnealingLRConfig):
        T_max = scheduler_config.T_max if scheduler_config.T_max is not None else total_epochs
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=scheduler_config.eta_min,
        )
    elif isinstance(scheduler_config, ReduceLROnPlateauConfig):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.mode,
            factor=scheduler_config.factor,
            patience=scheduler_config.patience,
            threshold=scheduler_config.threshold,
            threshold_mode=scheduler_config.threshold_mode,
            cooldown=scheduler_config.cooldown,
            min_lr=scheduler_config.min_lr,
        )
    elif isinstance(scheduler_config, WarmupCosineConfig):
        # Implement linear warmup followed by cosine annealing using SequentialLR
        warmup_epochs = scheduler_config.warmup_epochs
        T_max = (
            scheduler_config.T_max
            if scheduler_config.T_max is not None
            else total_epochs - warmup_epochs
        )

        # Linear warmup scheduler
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6,  # Start from very small LR
            end_factor=1.0,     # Reach full LR
            total_iters=warmup_epochs,
        )

        # Cosine annealing scheduler
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=scheduler_config.eta_min,
        )

        # Sequential scheduler: warmup then cosine
        return optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        raise ValueError(f"Unknown scheduler type: {type(scheduler_config)}")


class TemperatureAnnealer:
    """Manages temperature/sharpness annealing for SoftTree models.

    Temperature annealing gradually increases the sharpness parameter in SoftTree models
    during training, making the routing decisions progressively sharper and more deterministic.
    """

    def __init__(self, config: TemperatureAnnealingConfig, total_epochs: int):
        """Initialize temperature annealer.

        Args:
            config: Temperature annealing configuration
            total_epochs: Total number of training epochs
        """
        self.config = config
        self.total_epochs = total_epochs
        self.initial_temp = config.initial_temp
        self.final_temp = config.final_temp

    def get_temperature(self, epoch: int) -> float:
        """Compute temperature for current epoch.

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            Temperature value for this epoch
        """
        if self.config.schedule_type == "linear":
            return self._linear_schedule(epoch)
        elif self.config.schedule_type == "exponential":
            return self._exponential_schedule(epoch)
        elif self.config.schedule_type == "cosine":
            return self._cosine_schedule(epoch)
        elif self.config.schedule_type == "step":
            return self._step_schedule(epoch)
        else:
            raise ValueError(f"Unknown schedule type: {self.config.schedule_type}")

    def _linear_schedule(self, epoch: int) -> float:
        """Linear temperature increase from initial_temp to final_temp."""
        progress = min(epoch / max(self.total_epochs - 1, 1), 1.0)
        return self.initial_temp + progress * (self.final_temp - self.initial_temp)

    def _exponential_schedule(self, epoch: int) -> float:
        """Exponential temperature increase from initial_temp to final_temp."""
        if self.config.exp_gamma is None:
            raise ValueError("exp_gamma must be set for exponential schedule")

        # Compute the growth factor such that we reach final_temp at the last epoch
        # final_temp = initial_temp * (gamma ^ total_epochs)
        # This allows user-specified gamma, or we compute it
        gamma = self.config.exp_gamma
        progress = epoch / max(self.total_epochs - 1, 1)
        # Use logarithmic interpolation
        log_initial = np.log(self.initial_temp)
        log_final = np.log(self.final_temp)
        return float(np.exp(log_initial + progress * (log_final - log_initial)))

    def _cosine_schedule(self, epoch: int) -> float:
        """Cosine annealing temperature increase from initial_temp to final_temp."""
        progress = min(epoch / max(self.total_epochs - 1, 1), 1.0)
        # Cosine curve from 0 to 1
        cosine_progress = (1 - np.cos(progress * np.pi)) / 2
        return self.initial_temp + cosine_progress * (self.final_temp - self.initial_temp)

    def _step_schedule(self, epoch: int) -> float:
        """Step-based temperature increase."""
        if self.config.step_size is None or self.config.step_gamma is None:
            raise ValueError("step_size and step_gamma must be set for step schedule")

        num_steps = epoch // self.config.step_size
        return min(
            self.initial_temp * (self.config.step_gamma ** num_steps),
            self.final_temp
        )

    def update_model_temperature(self, model: ReGNN, epoch: int) -> float:
        """Update temperature/sharpness in model's SoftTree components.

        Args:
            model: ReGNN model to update
            epoch: Current epoch (0-indexed)

        Returns:
            The new temperature value applied
        """
        new_temp = self.get_temperature(epoch)

        # Navigate to index_prediction_model and update temperature
        if hasattr(model, 'index_prediction_model'):
            index_model = model.index_prediction_model
            if hasattr(index_model, 'use_soft_tree') and index_model.use_soft_tree:
                # Update temperature through the hierarchy
                if hasattr(index_model, 'set_temperature'):
                    index_model.set_temperature(new_temp)

        return new_temp


class BatchSizeScheduler:
    """Manages dynamic batch size scheduling during training.
    
    Supports two schedule types:
    - Step-based (StepBatchSizeConfig): batch_size = initial * gamma^(epoch // step_size)
    - Piecewise (PiecewiseBatchSizeConfig): arbitrary (epoch, batch_size) milestones
    
    Signals when DataLoader recreation is needed to accommodate the new batch size.
    """
    
    def __init__(self, config, initial_batch_size: int):
        """Initialize batch size scheduler.
        
        Args:
            config: StepBatchSizeConfig or PiecewiseBatchSizeConfig configuration
            initial_batch_size: Starting batch size for training (used as fallback
                for piecewise schedules before the first milestone)
        """
        self.config = config
        self.initial_batch_size = initial_batch_size
        self.current_epoch = 0

        # For piecewise config, the first milestone at epoch 0 (if present)
        # overrides the initial batch size
        from regnn.train.base import PiecewiseBatchSizeConfig
        if isinstance(config, PiecewiseBatchSizeConfig):
            first_epoch, first_bs = config.milestones[0]
            if first_epoch == 0:
                self.current_batch_size = first_bs
            else:
                self.current_batch_size = initial_batch_size
        else:
            self.current_batch_size = initial_batch_size
        
    def get_batch_size(self, epoch: int) -> int:
        """Compute batch size for given epoch.
        
        Args:
            epoch: Current epoch (0-indexed)
            
        Returns:
            Batch size for this epoch
        """
        from regnn.train.base import PiecewiseBatchSizeConfig
        
        if isinstance(self.config, PiecewiseBatchSizeConfig):
            return self._get_piecewise_batch_size(epoch)
        else:
            return self._get_step_batch_size(epoch)
    
    def _get_step_batch_size(self, epoch: int) -> int:
        """Step-based: batch_size = initial * gamma^(epoch // step_size)"""
        num_steps = epoch // self.config.step_size
        new_batch_size = self.initial_batch_size * (self.config.gamma ** num_steps)
        
        # Apply max_batch_size cap if specified
        if self.config.max_batch_size is not None:
            new_batch_size = min(new_batch_size, self.config.max_batch_size)
            
        return int(new_batch_size)
    
    def _get_piecewise_batch_size(self, epoch: int) -> int:
        """Piecewise: look up the last milestone at or before the current epoch."""
        milestones = self.config.milestones  # Already sorted by epoch from validator
        batch_size = self.initial_batch_size
        
        for milestone_epoch, milestone_bs in milestones:
            if milestone_epoch <= epoch:
                batch_size = milestone_bs
            else:
                break
                
        return int(batch_size)
    
    def step(self, epoch: int) -> Tuple[int, bool]:
        """Update epoch and return batch size and whether DataLoader needs recreation.
        
        Args:
            epoch: Current epoch (0-indexed)
            
        Returns:
            Tuple of (new_batch_size, should_recreate_dataloader)
        """
        self.current_epoch = epoch
        new_batch_size = self.get_batch_size(epoch)
        
        # Check if batch size has changed
        should_recreate = new_batch_size != self.current_batch_size
        
        if should_recreate:
            self.current_batch_size = new_batch_size
            
        return new_batch_size, should_recreate
    
    def should_update(self, epoch: int) -> bool:
        """Check if batch size should change at this epoch.
        
        Args:
            epoch: Current epoch (0-indexed)
            
        Returns:
            True if batch size changes at this epoch
        """
        current_bs = self.get_batch_size(epoch)
        if epoch == 0:
            return False
        previous_bs = self.get_batch_size(epoch - 1)
        return current_bs != previous_bs


class PerGroupScheduler:
    """Manages separate learning rate schedulers for different parameter groups.
    
    This wrapper allows different scheduler configurations for NN parameters and 
    regression parameters, enabling fine-grained control over learning rate decay.
    
    PyTorch schedulers operate on all parameter groups in an optimizer. This class
    works around that limitation by manually updating learning rates for specific
    parameter groups based on their individual scheduler configurations.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        scheduler_configs: Dict[str, Optional[SchedulerConfigUnion]],
        param_group_indices: Dict[str, int],
        total_epochs: int,
    ):
        """Initialize per-group scheduler.
        
        Args:
            optimizer: The optimizer containing all parameter groups
            scheduler_configs: Dictionary mapping group names to scheduler configs
                             e.g., {"nn": StepLRConfig(...), "regression": ExponentialLRConfig(...)}
            param_group_indices: Dictionary mapping group names to parameter group indices
                                e.g., {"nn": 0, "regression": 1}
            total_epochs: Total number of training epochs
        """
        self.optimizer = optimizer
        self.scheduler_configs = scheduler_configs
        self.param_group_indices = param_group_indices
        self.total_epochs = total_epochs
        
        # Store base learning rates for each group
        self.base_lrs = {}
        for group_name, idx in param_group_indices.items():
            self.base_lrs[group_name] = optimizer.param_groups[idx]['lr']
        
        # Track whether any scheduler is ReduceLROnPlateau
        self.is_reduce_on_plateau = any(
            isinstance(config, ReduceLROnPlateauConfig)
            for config in scheduler_configs.values()
            if config is not None
        )
        
        # Track current epoch for manual scheduling
        self.current_epoch = 0
        
    def step(self, metrics: Optional[float] = None):
        """Step all schedulers and update learning rates.
        
        Args:
            metrics: Optional metric value for ReduceLROnPlateau schedulers
        """
        # Increment epoch counter first (so epoch 0 -> 1 on first step)
        self.current_epoch += 1
        
        for group_name, config in self.scheduler_configs.items():
            if config is None:
                # No scheduler for this group, LR stays constant
                continue
                
            idx = self.param_group_indices[group_name]
            base_lr = self.base_lrs[group_name]
            
            # Compute new learning rate based on scheduler type
            new_lr = self._compute_lr(config, base_lr, metrics)
            
            # Update the parameter group's learning rate
            self.optimizer.param_groups[idx]['lr'] = new_lr
    
    def _compute_lr(
        self,
        config: SchedulerConfigUnion,
        base_lr: float,
        metrics: Optional[float] = None,
    ) -> float:
        """Compute learning rate for a given scheduler configuration.
        
        Args:
            config: Scheduler configuration
            base_lr: Base learning rate for this parameter group
            metrics: Optional metric value for ReduceLROnPlateau
            
        Returns:
            Computed learning rate for current epoch
        """
        epoch = self.current_epoch
        
        if isinstance(config, StepLRConfig):
            # LR = base_lr * gamma^(epoch // step_size)
            decay_factor = config.gamma ** (epoch // config.step_size)
            return base_lr * decay_factor
            
        elif isinstance(config, ExponentialLRConfig):
            # LR = base_lr * gamma^epoch
            return base_lr * (config.gamma ** epoch)
            
        elif isinstance(config, CosineAnnealingLRConfig):
            # Cosine annealing
            T_max = config.T_max if config.T_max is not None else self.total_epochs
            eta_min = config.eta_min
            
            # Standard cosine annealing formula
            import math
            return eta_min + (base_lr - eta_min) * (
                1 + math.cos(math.pi * epoch / T_max)
            ) / 2
            
        elif isinstance(config, ReduceLROnPlateauConfig):
            # ReduceLROnPlateau requires stateful tracking of patience/cooldown
            # This is complex to implement manually, so we'll use a PyTorch scheduler
            # but only for this specific parameter group
            # For now, return current LR (will be handled by a separate implementation)
            # TODO: Implement stateful ReduceLROnPlateau tracking
            return self.optimizer.param_groups[self.param_group_indices[
                list(self.scheduler_configs.keys())[
                    list(self.scheduler_configs.values()).index(config)
                ]
            ]]['lr']
            
        elif isinstance(config, WarmupCosineConfig):
            # Linear warmup followed by cosine annealing
            warmup_epochs = config.warmup_epochs
            
            if epoch < warmup_epochs:
                # Linear warmup: start from very small LR to base_lr
                start_factor = 1e-6
                return base_lr * (start_factor + (1.0 - start_factor) * epoch / warmup_epochs)
            else:
                # Cosine annealing after warmup
                T_max = (
                    config.T_max if config.T_max is not None
                    else self.total_epochs - warmup_epochs
                )
                eta_min = config.eta_min
                epoch_after_warmup = epoch - warmup_epochs
                
                import math
                return eta_min + (base_lr - eta_min) * (
                    1 + math.cos(math.pi * epoch_after_warmup / T_max)
                ) / 2
        else:
            raise ValueError(f"Unknown scheduler type: {type(config)}")


def setup_loss_and_optimizer(
    model: ReGNN,
    training_hyperparams: TrainingHyperParams,  # Use the specific type
) -> Tuple[
    nn.Module,
    Optional[nn.Module],
    optim.Optimizer,
    Optional[Union[
        optim.lr_scheduler.StepLR,
        optim.lr_scheduler.ExponentialLR,
        optim.lr_scheduler.CosineAnnealingLR,
        optim.lr_scheduler.ReduceLROnPlateau,
        optim.lr_scheduler.SequentialLR,
        "PerGroupScheduler",
    ]],
]:
    """Setup loss function, regularization, optimizer, and scheduler based on TrainingHyperParams.

    Returns:
        Tuple of (loss_function, regularization, optimizer, scheduler)
        
    Note:
        If per-group schedulers (scheduler_nn, scheduler_regression) are specified in 
        optimizer_config, a PerGroupScheduler instance is returned. Otherwise, a standard
        PyTorch scheduler is returned that applies to all parameter groups.
    """

    loss_opts = training_hyperparams.loss_options
    optimizer_opts = training_hyperparams.optimizer_config
    loss_func: nn.Module
    regularization: Optional[nn.Module] = None

    # 1. Setup Loss Function
    # Validation for KLDLossConfig compatibility with regnn_model_config (vae=True, output_mu_var=True)
    # is handled by MacroConfig's model_validator.
    if isinstance(loss_opts, KLDLossConfig):
        loss_func = vae_kld_regularized_loss(
            lambda_reg=loss_opts.lamba_reg,
            reduction=loss_opts.reduction,
        )
    elif isinstance(loss_opts, TreeLossConfig):
        # Tree routing regularization loss
        loss_func = tree_routing_regularized_loss(
            lambda_tree=loss_opts.lambda_tree,
            reduction=loss_opts.reduction,
        )
    elif isinstance(loss_opts, PriorPenaltyLossConfig):
        # Prior penalty (L2 on treatment effects) loss
        loss_func = prior_penalty_loss(
            lambda_prior=loss_opts.lambda_prior,
            reduction=loss_opts.reduction,
        )
    elif isinstance(loss_opts, MSELossConfig):
        loss_func = nn.MSELoss(reduction=loss_opts.reduction)
    else:
        # This case should ideally be prevented by Pydantic if using discriminated unions for LossConfigs subtypes.
        # If LossConfigs is a Union of MSELossConfig | KLDLossConfig | TreeLossConfig | PriorPenaltyLossConfig, this path might not be reachable
        # if the input `loss_options` is always one of the specific types.
        raise ValueError(
            f"Unsupported loss configuration type: {type(loss_opts)}. "
            f"Expected MSELossConfig, KLDLossConfig, TreeLossConfig, or PriorPenaltyLossConfig. Ensure training_hyperparams.loss_options is correctly initialized."
        )

    # 2. Setup Regularization
    if loss_opts.regularization:
        reg_config = loss_opts.regularization
        if isinstance(reg_config, ElasticNetRegConfig):
            regularization = elasticnet_loss(
                reduction=loss_opts.reduction, alpha=reg_config.elastic_net_alpha
            )
        elif reg_config.name == "lasso":
            # Assuming a LassoRegConfig would be similar if it existed formally
            # For now, relies on name and expects regularization_alpha from base RegularizationConfig
            regularization = lasso_loss(reduction=loss_opts.reduction)
        else:
            print(
                f"Warning: Unknown or non-specific regularization type specified: {reg_config.name}. No additive penalty will be applied beyond optimizer weight decay."
            )

    # 3. Setup Optimizer
    nn_params = [p for p in model.index_prediction_model.parameters()]
    if model.include_bias_focal_predictor:
        nn_params.append(model.xf_bias)
    optimizer = optim.AdamW(
        [
            {
                "params": nn_params,
                "weight_decay": optimizer_opts.weight_decay.weight_decay_nn,
                "lr": optimizer_opts.lr.lr_nn,
            },
            {
                "params": model.mmr_parameters,
                "weight_decay": optimizer_opts.weight_decay.weight_decay_regression,
                "lr": optimizer_opts.lr.lr_regression,
            },
        ],
        weight_decay=0.0,  # Top-level weight_decay is 0 as it's handled per param group
    )

    # 4. Setup Learning Rate Scheduler
    # Check if per-group schedulers are specified
    has_per_group = (
        optimizer_opts.scheduler_nn is not None or 
        optimizer_opts.scheduler_regression is not None
    )
    
    if has_per_group:
        # Use per-group schedulers (overrides global scheduler if both are set)
        scheduler_configs = {
            "nn": optimizer_opts.scheduler_nn,
            "regression": optimizer_opts.scheduler_regression,
        }
        # If only one per-group scheduler is set, use global scheduler for the other
        if optimizer_opts.scheduler_nn is None and optimizer_opts.scheduler is not None:
            scheduler_configs["nn"] = optimizer_opts.scheduler
        if optimizer_opts.scheduler_regression is None and optimizer_opts.scheduler is not None:
            scheduler_configs["regression"] = optimizer_opts.scheduler
            
        param_group_indices = {
            "nn": 0,  # First param group is NN parameters
            "regression": 1,  # Second param group is regression parameters
        }
        
        scheduler = PerGroupScheduler(
            optimizer=optimizer,
            scheduler_configs=scheduler_configs,
            param_group_indices=param_group_indices,
            total_epochs=training_hyperparams.epochs,
        )
    else:
        # Use global scheduler (backward compatible behavior)
        scheduler = create_lr_scheduler(
            optimizer,
            optimizer_opts.scheduler,
            training_hyperparams.epochs,
        )

    return loss_func, regularization, optimizer, scheduler


def balance_gradients_for_regnn(
    model: ReGNN, desired_ratio: float = 1.0, eps: float = 1e-8
):
    """
    Adjust gradient magnitudes for ReGNN model to prioritize modulation (index_prediction_model)
    vs. linear head learning (linear_weights and focal_predictor_main_weight).

    Parameters:
    - model: an instance of ReGNN
    - desired_ratio: target ratio of mod_head norm to linear_head norm (default = 1.0)
    - eps: small constant to avoid divide-by-zero
    """
    linear_norm = 0.0
    mod_norm = 0.0

    # First, compute current gradient norms
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if "linear_weights" in name or "focal_predictor_main_weight" in name:
            linear_norm += param.grad.norm(2).item() ** 2
        elif "index_prediction_model" in name:
            mod_norm += param.grad.norm(2).item() ** 2

    linear_norm = linear_norm**0.5
    mod_norm = mod_norm**0.5

    # Compute scaling factor for linear gradients
    scale = (mod_norm + eps) / (linear_norm + eps) * desired_ratio
    if scale > 1.0:
        return

    # Apply rescaling to linear gradients
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if "linear_weights" in name or "focal_predictor_main_weight" in name:
            param.grad.mul_(scale)


def format_epoch_printout(
    base_printout: str,
    current_epoch: int,
    probe_manager: Optional[Any] = None,
) -> str:
    """
    Formats and appends relevant information from probe results to a base printout string.
    Extracts both epoch-level and iteration-level results from the probe manager's trajectory.

    Args:
        base_printout: The initial printout string (e.g., epoch number and training loss).
        current_epoch: The current epoch number (0-indexed).
        probe_manager: Optional ProbeManager to access probe results from trajectory.

    Returns:
        An augmented string with information from recognized probes.
    """
    additional_info = []

    # Sort by data_source then probe_type for consistent ordering if multiple results exist
    # This is a simple sort, more complex might be needed if specific order is critical
    # For now, primarily to group by data_source for display
    # sorted_results = sorted(epoch_probe_results, key=lambda r: (r.data_source, r.probe_type_name))

    # It might be better to collect specific types of results first, then format them
    # e.g., all test objectives, then all test p-values

    test_objective_str: Optional[str] = None
    val_objective_str: Optional[str] = None  # If you have validation data source
    train_l2_norm_str: Optional[str] = None
    test_l2_norm_str: Optional[str] = None
    train_pval_str: Optional[str] = None
    test_pval_str: Optional[str] = None

    # Extract probe results from trajectory for current epoch
    if probe_manager is not None and hasattr(probe_manager, "trajectory"):
        from regnn.probe.dataclass.probe_config import FrequencyType

        epoch_results: List[ProbeData] = []
        iteration_pval_results: List[OLSModeratedResultsProbe] = []

        # Find all snapshots for current epoch
        for snapshot in probe_manager.trajectory.data:
            if snapshot.epoch == current_epoch and hasattr(
                snapshot, "frequency_context"
            ):
                if snapshot.frequency_context == FrequencyType.EPOCH:
                    # Collect epoch-level results
                    epoch_results.extend(snapshot.measurements)
                elif snapshot.frequency_context == FrequencyType.ITERATION:
                    # Collect iteration-level regression results
                    for measurement in snapshot.measurements:
                        if isinstance(measurement, OLSModeratedResultsProbe):
                            iteration_pval_results.append(measurement)

        # Process epoch-level results first
        for result in epoch_results:
            if isinstance(result, ObjectiveProbe):
                ds_str = result.data_source.upper()
                obj_name = result.objective_name or "Objective"
                obj_val_str = (
                    f"{result.objective:.4f}"
                    if isinstance(result.objective, float)
                    else str(result.objective)
                )

                if ds_str == DataSource.TEST.value:
                    test_objective_str = f"Test {obj_name}: {obj_val_str}"
                elif ds_str == DataSource.VALIDATION.value:
                    val_objective_str = f"Val {obj_name}: {obj_val_str}"

            elif isinstance(result, OLSModeratedResultsProbe):
                ds_str = result.data_source.upper()
                pval_str = (
                    f"{result.interaction_pval:.4f}"
                    if result.interaction_pval is not None
                    else "N/A"
                )
                r2_str = (
                    f"{result.rsquared:.3f}" if result.rsquared is not None else "N/A"
                )

                current_pval_info = f"{ds_str} PVal: {pval_str} (R2: {r2_str})"
                if ds_str == DataSource.TRAIN.value:
                    train_pval_str = current_pval_info
                elif ds_str == DataSource.TEST.value:
                    test_pval_str = current_pval_info

            elif isinstance(result, L2NormProbe):
                ds_str = result.data_source.upper()
                main_norm_str = (
                    f"{result.main_norm:.4e}"
                    if isinstance(result.main_norm, float)
                    else str(result.main_norm)
                )
                index_norm_str = (
                    f"{result.index_norm:.4e}"
                    if isinstance(result.index_norm, float)
                    else str(result.index_norm)
                )
                l2_info = f"{ds_str} L2: Main={main_norm_str}, Index={index_norm_str}"
                if ds_str == DataSource.TRAIN.value:
                    train_l2_norm_str = l2_info
                elif ds_str == DataSource.TEST.value:
                    test_l2_norm_str = l2_info
                elif result.data_source.upper() == DataSource.ALL.value:
                    train_l2_norm_str = (
                        f"Global L2: Main={main_norm_str}, Index={index_norm_str}"
                    )

        # Get last iteration results for each data source (only if no epoch-level results exist)
        last_iteration_results: Dict[str, OLSModeratedResultsProbe] = {}
        for result in iteration_pval_results:
            ds_key = result.data_source.upper()
            last_iteration_results[ds_key] = (
                result  # Later ones will overwrite earlier ones
            )

        # Add iteration-level results only if no epoch-level results exist for that data source
        for ds_key, result in last_iteration_results.items():
            pval_str = (
                f"{result.interaction_pval:.4f}"
                if result.interaction_pval is not None
                else "N/A"
            )
            r2_str = f"{result.rsquared:.3f}" if result.rsquared is not None else "N/A"
            current_pval_info = f"{ds_key} PVal: {pval_str} (R2: {r2_str}) [iter]"

            if ds_key == DataSource.TRAIN.value and train_pval_str is None:
                train_pval_str = current_pval_info
            elif ds_key == DataSource.TEST.value and test_pval_str is None:
                test_pval_str = current_pval_info

    # Assemble the printout string in a preferred order
    if test_objective_str:
        additional_info.append(test_objective_str)
    if val_objective_str:
        additional_info.append(val_objective_str)
    if train_pval_str:
        additional_info.append(train_pval_str)
    if test_pval_str:
        additional_info.append(test_pval_str)
    if (
        train_l2_norm_str and DataSource.ALL.value not in train_l2_norm_str
    ):  # Avoid double print if global was captured
        additional_info.append(train_l2_norm_str)
    elif train_l2_norm_str:  # If it's the global one
        additional_info.insert(0, train_l2_norm_str)  # Put global L2 first

    if test_l2_norm_str:
        additional_info.append(test_l2_norm_str)

    if additional_info:
        return base_printout + " | " + " | ".join(additional_info)
    return base_printout
