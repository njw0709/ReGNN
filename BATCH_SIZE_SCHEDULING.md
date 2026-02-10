# Batch Size Scheduling

## Overview

Batch size scheduling allows you to dynamically increase the batch size during training. This feature can improve training stability, convergence speed, and generalization performance by starting with smaller batches (better gradient estimates) and gradually increasing to larger batches (better computational efficiency).

## Features

- **Step-based scheduling**: Increase batch size at regular epoch intervals
- **Configurable growth rate**: Control how aggressively batch size increases
- **Maximum batch size cap**: Prevent out-of-memory errors with optional upper bound
- **Automatic DataLoader recreation**: Training loop handles DataLoader updates seamlessly
- **Type-safe configuration**: Pydantic validation ensures correct parameters

## Benefits of Batch Size Scheduling

1. **Improved Generalization**: Starting with smaller batches provides noisier gradient estimates, which can help escape sharp minima and find flatter solutions that generalize better.

2. **Training Stability**: Small batches early in training help avoid instabilities when the model is still learning basic patterns.

3. **Computational Efficiency**: Larger batches later in training improve GPU utilization and training speed when fine-tuning.

4. **Memory Management**: Gradually increasing batch size lets you start training even with limited GPU memory, then utilize more memory as gradients stabilize.

## Usage

### Basic Example

```python
from regnn.train import (
    TrainingHyperParams,
    StepBatchSizeConfig,
    OptimizerConfig,
)

# Configure batch size scheduler
batch_scheduler_config = StepBatchSizeConfig(
    step_size=20,        # Increase batch size every 20 epochs
    gamma=2,             # Multiply batch size by 2
    max_batch_size=2048  # Cap at 2048 (optional)
)

# Use in training configuration
training_hp = TrainingHyperParams(
    epochs=100,
    batch_size=256,  # Initial batch size
    batch_size_scheduler=batch_scheduler_config,
    optimizer_config=OptimizerConfig(...),
)
```

With this configuration:
- Epochs 0-19: batch_size = 256
- Epochs 20-39: batch_size = 512
- Epochs 40-59: batch_size = 1024
- Epochs 60-79: batch_size = 2048 (hits max_batch_size)
- Epochs 80+: batch_size = 2048 (stays at max)

### Conservative Growth

For a more gradual increase:

```python
batch_scheduler_config = StepBatchSizeConfig(
    step_size=10,        # Increase every 10 epochs
    gamma=2,             # Double each time
    max_batch_size=1024
)

training_hp = TrainingHyperParams(
    epochs=50,
    batch_size=128,
    batch_size_scheduler=batch_scheduler_config,
)
```

Growth pattern:
- Epochs 0-9: batch_size = 128
- Epochs 10-19: batch_size = 256
- Epochs 20-29: batch_size = 512
- Epochs 30+: batch_size = 1024

### Aggressive Growth

For rapid batch size increase:

```python
batch_scheduler_config = StepBatchSizeConfig(
    step_size=15,
    gamma=4,  # Quadruple each step
    max_batch_size=4096
)

training_hp = TrainingHyperParams(
    epochs=60,
    batch_size=64,
    batch_size_scheduler=batch_scheduler_config,
)
```

Growth pattern:
- Epochs 0-14: batch_size = 64
- Epochs 15-29: batch_size = 256
- Epochs 30-44: batch_size = 1024
- Epochs 45+: batch_size = 4096

### Without Batch Size Scheduling

Batch size scheduling is optional. Simply omit the `batch_size_scheduler` parameter:

```python
training_hp = TrainingHyperParams(
    epochs=100,
    batch_size=512,  # Constant batch size throughout training
)
```

## Configuration Reference

### StepBatchSizeConfig

```python
class StepBatchSizeConfig(BaseModel):
    type: Literal["step"] = "step"
    step_size: int  # Period of batch size increase (in epochs), must be > 0
    gamma: int      # Multiplicative factor, must be >= 2
    max_batch_size: Optional[int] = None  # Optional upper bound, must be > 0
```

**Parameters**:

- **`step_size`** (required): Number of epochs between batch size increases. Must be a positive integer.
  
- **`gamma`** (required): Factor by which to multiply the batch size at each step. Must be an integer >= 2.
  
- **`max_batch_size`** (optional): Maximum batch size cap. If specified, batch size will never exceed this value. Useful for preventing out-of-memory errors.

**Formula**:

```
batch_size(epoch) = min(
    initial_batch_size * gamma^(epoch // step_size),
    max_batch_size  # if specified
)
```

## Integration with Training Loop

The batch size scheduler integrates seamlessly with the training loop:

1. **Initialization**: The scheduler is created before training begins
2. **Epoch Start**: At the start of each epoch, the scheduler checks if batch size should change
3. **DataLoader Recreation**: If batch size changes, the train DataLoader is automatically recreated
4. **Logging**: Batch size changes are printed to stdout for visibility

Example output:
```
Epoch 20: Updating batch size from 256 to 512
Epoch 40: Updating batch size from 512 to 1024
Epoch 60: Updating batch size from 1024 to 2048
```

## Best Practices

### 1. Start Small, End Large

A typical pattern is to start with a smaller batch size (128-512) and gradually increase to a larger batch size (1024-4096):

```python
StepBatchSizeConfig(
    step_size=25,
    gamma=2,
    max_batch_size=2048
)
```

### 2. Coordinate with Learning Rate

When increasing batch size, you may want to adjust learning rate accordingly. Consider using per-group schedulers to fine-tune learning rates:

```python
training_hp = TrainingHyperParams(
    batch_size=128,
    batch_size_scheduler=StepBatchSizeConfig(step_size=20, gamma=2),
    optimizer_config=OptimizerConfig(
        lr=LearningRateConfig(lr_nn=0.001, lr_regression=0.02),
        # Optionally decay learning rate as batch size increases
        scheduler_nn=ExponentialLRConfig(gamma=0.95),
    ),
)
```

### 3. Monitor GPU Memory

Use `max_batch_size` to prevent out-of-memory errors:

```python
# If you know your GPU can handle up to 4096 batch size
StepBatchSizeConfig(
    step_size=15,
    gamma=2,
    max_batch_size=4096
)
```

### 4. Align with Training Phases

Consider aligning batch size increases with other training phases:

```python
# 100 epochs total
# Increase batch size every 25 epochs (4 phases)
StepBatchSizeConfig(
    step_size=25,
    gamma=2,
    max_batch_size=None
)
```

### 5. Test Your Configuration

Run a short training session to verify your configuration works before committing to a long training run:

```python
# Test with a small number of epochs first
test_hp = TrainingHyperParams(
    epochs=10,  # Just to test
    batch_size=256,
    batch_size_scheduler=StepBatchSizeConfig(step_size=3, gamma=2),
)
```

## Comparison with Other Techniques

### vs. Constant Batch Size

**Batch Size Scheduling**:
- Better generalization (smaller batches early)
- More stable training (gradual increase)
- Better GPU utilization (larger batches later)

**Constant Batch Size**:
- Simpler to configure
- More predictable training behavior
- Easier to reproduce

### vs. Learning Rate Warmup

Batch size scheduling and learning rate warmup complement each other:

- **LR Warmup**: Gradually increases learning rate at the start of training
- **Batch Size Scheduling**: Gradually increases batch size throughout training

You can use both together:

```python
training_hp = TrainingHyperParams(
    batch_size=256,
    batch_size_scheduler=StepBatchSizeConfig(step_size=20, gamma=2),
    optimizer_config=OptimizerConfig(
        lr=LearningRateConfig(lr_nn=0.001),
        scheduler_nn=WarmupCosineConfig(warmup_epochs=10, T_max=100),
    ),
)
```

## Common Use Cases

### 1. Large Dataset, Limited GPU Memory

Start with a batch size you can fit in memory, then increase:

```python
StepBatchSizeConfig(
    step_size=15,
    gamma=2,
    max_batch_size=1024  # Your GPU limit
)
```

### 2. Fine-Tuning or Transfer Learning

Use smaller batches initially for better gradient estimates:

```python
StepBatchSizeConfig(
    step_size=10,
    gamma=2,
    max_batch_size=512
)
```

### 3. Very Long Training Runs

Gradually increase batch size over hundreds of epochs:

```python
StepBatchSizeConfig(
    step_size=50,  # Increase every 50 epochs
    gamma=2,
    max_batch_size=4096
)
```

## Testing

Comprehensive unit tests are available in `tests/unit/train/test_batch_size_scheduler.py`:

```bash
uv run pytest tests/unit/train/test_batch_size_scheduler.py -v
```

Tests cover:
- Step schedule with different intervals
- Maximum batch size cap enforcement
- Epoch tracking
- DataLoader recreation signaling
- Pydantic validation
- Integration with `TrainingHyperParams`

## Implementation Details

### BatchSizeScheduler Class

The `BatchSizeScheduler` class (in `regnn/macroutils/utils.py`) manages batch size updates:

```python
class BatchSizeScheduler:
    def __init__(self, config: StepBatchSizeConfig, initial_batch_size: int)
    def get_batch_size(self, epoch: int) -> int
    def step(self, epoch: int) -> Tuple[int, bool]
    def should_update(self, epoch: int) -> bool
```

### DataLoader Recreation

When batch size changes, the training loop recreates the train DataLoader:

```python
if batch_size_scheduler:
    new_batch_size, should_recreate = batch_size_scheduler.step(epoch)
    if should_recreate:
        train_dataloader = TorchDataLoader(
            train_dataset,
            batch_size=new_batch_size,
            shuffle=training_hp.shuffle,
        )
```

This ensures:
- Correct batch sampling with new batch size
- Shuffle state is properly managed
- No memory leaks from old DataLoader

## Migration Guide

### Adding Batch Size Scheduling to Existing Training

**Before:**
```python
training_hp = TrainingHyperParams(
    epochs=100,
    batch_size=512,
    optimizer_config=OptimizerConfig(...),
)
```

**After:**
```python
training_hp = TrainingHyperParams(
    epochs=100,
    batch_size=256,  # Start smaller
    batch_size_scheduler=StepBatchSizeConfig(
        step_size=20,
        gamma=2,
        max_batch_size=2048
    ),
    optimizer_config=OptimizerConfig(...),
)
```

## Future Extensions

Potential future enhancements:
- `LinearBatchSizeConfig`: Linear growth instead of exponential
- `ExponentialBatchSizeConfig`: Smooth exponential growth
- `CosineBatchSizeConfig`: Cosine annealing schedule
- Custom batch size functions via callbacks

## References

- Smith, S. L., et al. (2017). "Don't Decay the Learning Rate, Increase the Batch Size"
- Goyal, P., et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
- You, Y., et al. (2019). "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"

## Examples

See `scripts/train_simulated.py` for commented examples showing how to use batch size scheduling in practice.

## Support

For issues or questions about batch size scheduling:
1. Check this documentation
2. Review the test suite for examples
3. Examine `scripts/train_simulated.py` for usage patterns
4. Check GPU memory usage if hitting OOM errors
