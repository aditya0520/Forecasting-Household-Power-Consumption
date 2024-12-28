import torch.nn as nn

def combined_loss(predictions, targets, alpha=0.5):

    mse_loss = nn.MSELoss()(predictions, targets)
    mae_loss = nn.L1Loss()(predictions, targets)
    return alpha * mse_loss + (1 - alpha) * mae_loss