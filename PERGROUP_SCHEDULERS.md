# Per-Parameter-Group Learning Rate Schedulers

## Overview

This implementation adds support for independent learning rate schedules for neural network (NN) weights and linear/regression weights in ReGNN models. This allows fine-grained control over how different parts of the model learn during training.

## Features

- **Per-group schedulers**: Configure different LR schedules for NN parameters vs regression parameters
- **Backward compatible**: Existing code using a global `scheduler` continues to work unchanged
- **All scheduler types supported**: StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, WarmupCosine
- **Mix and match**: Use any combination of schedulers for different parameter groups

## Usage

### Basic Example: Per-Group Schedulers

```python
from regnn.train import (
    OptimizerConfig,
    LearningRateConfig,
    WeightDecayConfig,
    StepLRConfig,
    CosineAnnealingLRConfig,
)

optimizer_config = OptimizerConfig(
    lr=LearningRateConfig(lr_nn=0.001, lr_regression=0.02),
    weight_decay=WeightDecayConfig(weight_decay_nn=0.02, weight_decay_regression=0.0),
    # Different schedulers for each parameter group
    scheduler_nn=CosineAnnealingLRConfig(T_max=100, eta_min=1e-6),
    scheduler_regression=StepLRConfig(step_size=20, gamma=0.5),
)
```

In this example:
- NN weights use cosine annealing (smooth decay)
- Regression weights use step decay (periodic sharp drops)

### Backward Compatible: Global Scheduler

Existing code continues to work unchanged:

```python
optimizer_config = OptimizerConfig(
    lr=LearningRateConfig(lr_nn=0.001, lr_regression=0.02),
    # Global scheduler applies to both parameter groups
    scheduler=StepLRConfig(step_size=10, gamma=0.5),
)
```

### Mixed Configuration

You can specify a global scheduler with overrides:

```python
optimizer_config = OptimizerConfig(
    lr=LearningRateConfig(lr_nn=0.001, lr_regression=0.02),
    scheduler=StepLRConfig(step_size=10, gamma=0.5),  # Default for both
    scheduler_nn=CosineAnnealingLRConfig(T_max=100),   # Override for NN only
    # Regression will use the global scheduler
)
```

Note: If both global and per-group schedulers are set, a warning is issued and per-group schedulers take precedence.

### No Scheduler for One Group

You can disable scheduling for a specific group:

```python
optimizer_config = OptimizerConfig(
    lr=LearningRateConfig(lr_nn=0.001, lr_regression=0.02),
    scheduler_nn=ExponentialLRConfig(gamma=0.95),  # NN decays
    scheduler_regression=None,  # Regression stays constant
)
```

## Available Scheduler Types

All standard PyTorch schedulers are supported:

1. **StepLRConfig**: Decay LR by `gamma` every `step_size` epochs
2. **ExponentialLRConfig**: Multiply LR by `gamma` every epoch
3. **CosineAnnealingLRConfig**: Cosine annealing from initial LR to `eta_min`
4. **ReduceLROnPlateauConfig**: Reduce LR when a metric plateaus (note: metric-based)
5. **WarmupCosineConfig**: Linear warmup followed by cosine annealing

## Common Use Cases

### 1. Conservative NN Training, Aggressive Regression Fitting

```python
scheduler_nn=ExponentialLRConfig(gamma=0.98),  # Slow decay
scheduler_regression=StepLRConfig(step_size=10, gamma=0.5),  # Faster decay
```

### 2. Warmup for NN, Constant LR for Regression

```python
scheduler_nn=WarmupCosineConfig(warmup_epochs=5, T_max=100),
scheduler_regression=None,  # Keep constant
```

### 3. Same Schedule, Different Parameters

```python
# Both use StepLR but at different intervals
scheduler_nn=StepLRConfig(step_size=15, gamma=0.5),
scheduler_regression=StepLRConfig(step_size=30, gamma=0.5),
```

## Implementation Details

### PerGroupScheduler Class

The `PerGroupScheduler` class (in `regnn/macroutils/utils.py`) manages separate schedulers for each parameter group. It manually computes learning rates for each group based on their individual scheduler configurations.

### Parameter Groups

ReGNN models have two parameter groups:
- **Group 0 (NN)**: `index_prediction_model` parameters + optional `xf_bias`
- **Group 1 (Regression)**: MMR parameters (linear weights, focal predictor weight, intercept)

### Training Loop Integration

The training loop automatically detects `PerGroupScheduler` and handles it appropriately, including support for metric-based schedulers like ReduceLROnPlateau.

## Testing

Comprehensive tests are available in `tests/unit/train/test_per_group_schedulers.py`:

```bash
# Note: Tests require fixing numpy compatibility issues in the environment
uv run pytest tests/unit/train/test_per_group_schedulers.py -v
```

Tests cover:
- Different scheduler types per group
- Mixed configurations
- Validation warnings
- Epoch tracking
- Base LR storage
- All scheduler types (Step, Exponential, Cosine, WarmupCosine)

## Examples

See `scripts/train_simulated.py` for commented examples of how to use per-group schedulers in practice.

## Migration Guide

### From Global Scheduler to Per-Group

**Before:**
```python
optimizer_config=OptimizerConfig(
    lr=LearningRateConfig(lr_nn=0.001, lr_regression=0.02),
    scheduler=StepLRConfig(step_size=20, gamma=0.5),
)
```

**After (same behavior):**
```python
optimizer_config=OptimizerConfig(
    lr=LearningRateConfig(lr_nn=0.001, lr_regression=0.02),
    scheduler=StepLRConfig(step_size=20, gamma=0.5),  # Keep as-is
)
```

**After (different schedules):**
```python
optimizer_config=OptimizerConfig(
    lr=LearningRateConfig(lr_nn=0.001, lr_regression=0.02),
    scheduler_nn=CosineAnnealingLRConfig(T_max=100),
    scheduler_regression=StepLRConfig(step_size=20, gamma=0.5),
)
```

## Files Modified

1. **regnn/train/base.py**: Added `scheduler_nn` and `scheduler_regression` fields to `OptimizerConfig`
2. **regnn/macroutils/utils.py**: Added `PerGroupScheduler` class and updated `setup_loss_and_optimizer()`
3. **regnn/macroutils/trainer.py**: Updated training loop to handle `PerGroupScheduler`
4. **tests/unit/train/test_per_group_schedulers.py**: New comprehensive test suite
5. **scripts/train_simulated.py**: Added example usage in comments
