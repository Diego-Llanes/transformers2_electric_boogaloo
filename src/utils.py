import torch.nn.functional as F


def _classification_loss(preds, y):
    B, T, V = preds.shape
    return F.cross_entropy(preds.view(B*T, V), y.view(B*T))


def _regression_loss(preds, y):
    return F.mse_loss(preds, y)


TASK_TO_OBJECTIVE_FN = {
    'classification': _classification_loss,
    'regression': _regression_loss,
}
