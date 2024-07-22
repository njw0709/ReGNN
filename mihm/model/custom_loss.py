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


def vae_kld_regularized_loss(lambda_reg=0.01):
    def loss(y_true, y_pred_mean, int_pred_mu, int_pred_logvar, lambda_reg=0.01):
        kld_loss = -0.5 * torch.mean(
            1 + int_pred_logvar - int_pred_mu**2 - torch.exp(int_pred_logvar)
        )
        mse_loss = torch.mean((y_true - y_pred_mean) ** 2)
        total_loss = mse_loss + lambda_reg * kld_loss
        return total_loss

    return partial(loss, lambda_reg=lambda_reg)
