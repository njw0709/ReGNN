import torch
from functools import partial


def vae_kld_regularized_loss(lambda_reg=0.01, reduction: str = "mean"):
    def loss(
        y_true,
        y_pred_mean,
        int_pred_mu,
        int_pred_logvar,
        lambda_reg=0.01,
        reduction: str = reduction,
    ):
        kld_loss = -0.5 * (
            1 + int_pred_logvar - int_pred_mu**2 - torch.exp(int_pred_logvar)
        )
        mse_loss = (y_true - y_pred_mean) ** 2

        if reduction == "mean":
            total_loss = torch.mean(mse_loss) + lambda_reg * torch.mean(kld_loss)
        elif reduction == "sum":
            total_loss = torch.sum(mse_loss) + lambda_reg * torch.sum(kld_loss)
        elif reduction == "none":
            total_loss = mse_loss + lambda_reg * kld_loss
        else:
            raise NameError("reduction must be one of: sum, mean, none!")
        return total_loss

    return partial(loss, lambda_reg=lambda_reg)


def lasso_loss(reduction: str = "sum"):
    def loss(var):
        if reduction == "sum":
            return torch.sum(torch.abs(var))
        elif reduction == "mean":
            return torch.mean(torch.abs(var))
        elif reduction == "none":
            return torch.abs(var)
        else:
            raise NameError("reduction must be one of: sum, mean, none!")

    return loss


def elasticnet_loss(alpha=0.1, reduction: str = "mean"):
    def loss(var):
        if reduction == "sum":
            return (1.0 - alpha) * torch.sum(torch.square(var)) + alpha * torch.sum(
                torch.abs(var)
            )
        elif reduction == "mean":
            return (1.0 - alpha) * torch.mean(torch.square(var)) + alpha * torch.mean(
                torch.abs(var)
            )
        elif reduction == "none":
            return (1.0 - alpha) * torch.square(var) + alpha * torch.abs(var)
        else:
            raise NameError("reduction must be one of: sum, mean, none!")

    return loss


def ridge_loss(reduction: str = "sum"):
    def loss(var):
        if reduction == "sum":
            return torch.sum(torch.square(var))
        elif reduction == "mean":
            return torch.mean(torch.square(var))
        elif reduction == "none":
            return torch.square(var)
        else:
            raise NameError("reduction must be one of: sum, mean, none!")

    return loss


def tree_routing_regularized_loss(lambda_tree=0.01, reduction: str = "mean"):
    """Create a loss function that combines MSE with tree routing regularization.

    Similar to vae_kld_regularized_loss, this returns a partially applied loss function.
    Implements the Frosst & Hinton regularization that encourages balanced 50/50 routing.

    Args:
        lambda_tree: Weight for tree routing regularization
        reduction: How to reduce the loss ('mean', 'sum', or 'none')

    Returns:
        A loss function that takes (y_pred, y_true, moderators, model)
    """

    def loss(
        y_pred, y_true, moderators, model, lambda_tree=lambda_tree, reduction=reduction
    ):
        """Combined MSE + tree routing regularization loss.

        Args:
            y_pred: Model predictions
            y_true: Ground truth targets
            moderators: Input moderators (needed to compute routing regularization)
            model: ReGNN model instance (to access tree structure)
        """
        # Compute MSE loss
        mse_loss = (y_pred - y_true) ** 2
        # Compute tree routing regularization if applicable
        tree_reg = torch.tensor(0.0, device=y_pred.device)
        if hasattr(model, "index_prediction_model"):
            index_model = model.index_prediction_model
            if hasattr(index_model, "use_soft_tree") and index_model.use_soft_tree:
                # Compute regularization based on model structure
                if index_model.num_models == 1:
                    backbone = index_model.mlp
                    if hasattr(backbone, "compute_routing_regularization"):
                        tree_reg = backbone.compute_routing_regularization(moderators)
                else:
                    # Multiple models case
                    for i, backbone in enumerate(index_model.mlp):
                        if hasattr(backbone, "compute_routing_regularization"):
                            tree_reg = (
                                tree_reg
                                + backbone.compute_routing_regularization(moderators[i])
                            )
        # Combine losses
        if reduction == "mean":
            total_loss = torch.mean(mse_loss) + lambda_tree * tree_reg
        elif reduction == "sum":
            total_loss = torch.sum(mse_loss) + lambda_tree * tree_reg
        elif reduction == "none":
            # For 'none', add regularization as scalar to each element
            total_loss = mse_loss + (lambda_tree * tree_reg / mse_loss.numel())
        else:
            raise NameError("reduction must be one of: sum, mean, none!")
        return total_loss

    return partial(loss, lambda_tree=lambda_tree, reduction=reduction)
