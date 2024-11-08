import torch
from functools import partial


def var_adjusted_mse_loss(epsilon=1e-6, lambda_reg=0.01):
    def loss(y_true, y_pred_mean, y_pred_logvar, epsilon=1e-6, lambda_reg=0.01):
        variance = torch.exp(y_pred_logvar) + epsilon
        mse_loss = torch.mean(((y_true - y_pred_mean) ** 2) / variance)
        variance_regularization = torch.mean(torch.log(variance))
        total_loss = mse_loss + lambda_reg * variance_regularization
        return total_loss

    return partial(loss, epsilon=epsilon, lambda_reg=lambda_reg)


def vae_kld_regularized_loss(lambda_reg=0.01, reduction: str ="mean"):
    def loss(y_true, y_pred_mean, int_pred_mu, int_pred_logvar, lambda_reg=0.01, reduction: str = reduction):
        kld_loss = -0.5*(1 + int_pred_logvar - int_pred_mu**2 - torch.exp(int_pred_logvar))
        mse_loss = (y_true - y_pred_mean) ** 2

        if reduction=="mean":
            total_loss = torch.mean(mse_loss) + lambda_reg*torch.mean(kld_loss) 
        else:
            total_loss = mse_loss + lambda_reg * kld_loss
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
            return (1.0-alpha) * torch.sum(torch.square(var)) + alpha * torch.sum(torch.abs(var))
        elif reduction == "mean":
            return (1.0-alpha) * torch.mean(torch.square(var)) + alpha * torch.mean(torch.abs(var))
        elif reduction == "none":
            return (1.0-alpha) * torch.square(var) + alpha * torch.abs(var)
        else:
            raise NameError()
    return loss