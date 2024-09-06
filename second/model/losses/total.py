import torch
from typing import Tuple
from torch import Tensor, nn, DeviceObjType
from torch.nn import BCELoss, SmoothL1Loss
from focal_loss.focal_loss import FocalLoss
from second.model.losses.sine import SineSmoothL1Loss


class TotalLoss(nn.Module):
    """
    Implements the total loss described in SECOND's paper.

    Args:
        device (DeviceObjType): the device on which tensors should be.
        beta_1 (float, optional): constant coefficient for controlling the weight of classification loss.
        Defaults to 1.0.
        beta_2 (float, optional): constant coefficient for controlling the weight of regression losses.
        Defaults to 2.0.
        beta_3 (float, optional): constant coefficient for controlling the weight of direction classification loss.
        Defaults to 0.2.
    """

    def __init__(
        self, device: DeviceObjType, beta_1: float = 1.0, beta_2: float = 2.0, beta_3: float = 0.2
    ) -> None:
        super(TotalLoss, self).__init__()
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.weights = torch.FloatTensor([0.75, 0.25, 0.25, 0.75]).to(device)

        self.cls_loss = FocalLoss(gamma=2.0, weights=self.weights, eps=1e-6)
        self.reg_angle_loss = SineSmoothL1Loss()
        self.reg_loss = SmoothL1Loss()
        self.dir_loss = BCELoss()

    def forward(
        self,
        score: Tensor,
        reg: Tensor,
        drc: Tensor,
        class_pos: Tensor,
        targets: Tensor,
        directions: Tensor
    ) -> Tuple[Tensor]:
        """
        Combines all forward functions of required losses.

        Args:
            score (Tensor): The score prediction.
            reg (Tensor): The regression prediction.
            drc (Tensor): The direction prediction.
            class_pos (Tensor): The tensor that precises a class' presence.
            targets (Tensor): The learning target of the prediction.
            directions (Tensor): The encoded directions.

        Returns:
            Tuple[Tensor]: Computed losses.
        """
        score = score.permute(0, 2, 3, 1).contiguous()
        reg = reg.permute(0, 2, 3, 1).contiguous()
        drc = drc.permute(0, 2, 3, 1).contiguous()
        
        reg = reg.view(reg.size(0), reg.size(1), reg.size(2), -1, 7)
        targets = targets.view(targets.size(0), targets.size(1), targets.size(2), -1, 7)
        drc = drc.view(drc.size(0), drc.size(1), drc.size(2), -1, 2)

        focal_loss = self.cls_loss(score, class_pos.long())
        reg_loss = self.reg_loss(reg, targets)
        angle_loss = self.reg_angle_loss(torch.sin(reg[..., -1]), torch.sin(targets[..., -1]))
        dir_loss = self.dir_loss(drc, directions)

        total_loss = self.beta_1 * focal_loss + self.beta_2 * (angle_loss + reg_loss) + self.beta_3 * dir_loss

        return total_loss, focal_loss, reg_loss, dir_loss
