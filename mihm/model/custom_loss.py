import torch


def var_adjusted_mse_loss(
    y_true, y_pred_mean, y_pred_logvar, epsilon=1e-6, lambda_reg=0.01
):
    variance = torch.exp(y_pred_logvar) + epsilon
    mse_loss = torch.mean(((y_true - y_pred_mean) ** 2) / variance)
    variance_regularization = torch.mean(torch.log(variance))
    total_loss = mse_loss + lambda_reg * variance_regularization
    return total_loss
