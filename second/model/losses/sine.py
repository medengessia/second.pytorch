import torch
from torch import nn, Tensor
from mmdet.models.losses.utils import weighted_loss


@weighted_loss
def sine_smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.0) -> Tensor:
    """
    Computes the sine smooth L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        Tensor: Calculated loss.
    """
    assert beta > 0
    assert target.numel() > 0
    assert pred.size() == target.size(), 'The size of pred ' \
        f'{pred.size()} and target {target.size()} ' \
        'are inconsistent.'
    diff = torch.abs(torch.sin(pred - target))

    return torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)


class SineSmoothL1Loss(nn.Module):
    """
    Implements the sine smooth L1 loss function used in SECOND.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Defaults to 'mean'.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
    """
    def __init__(self, reduction: str = 'mean', beta: float = 1.0) -> None:
        super(SineSmoothL1Loss, self).__init__()
        self.reduction = reduction
        self.beta = beta
        
    def forward(self, pred: Tensor, target: Tensor, **kwargs) -> Tensor:
        """
        Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.

        Returns:
            Tensor: Calculated loss.
        """        
        return sine_smooth_l1_loss(pred, target, beta=self.beta, reduction=self.reduction, **kwargs)