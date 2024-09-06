import torch
import torch.nn.functional as F

from typing import Tuple
from torch import Tensor, nn


# conv2d + bn + relu
class Conv2d(nn.Module):

    def __init__(self,in_channels,out_channels,k,s,p, activation=True, batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=k,stride=s,padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation
    def forward(self,x):
        x = self.conv(x)
        if self.bn is not None:
            x=self.bn(x)
        if self.activation:
            return F.relu(x,inplace=True)
        else:
            return x

class RPN(nn.Module):
    """
    Implements the Region Proposal Network used in SECOND.

    Args:
        num_features (int): Input dimension coming from Sparse Convolutional layers.
        num_classes (int): The number of classes.
        num_directions (int): The number of directions.
        num_regression_offsets (int): The number of regression offsets.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        num_directions: int,
        num_regression_offsets: int,
    ) -> None:
        super(RPN, self).__init__()

        self.num_classes = num_classes
        self.num_directions = num_directions
        self.num_regression_offsets = num_regression_offsets
        self.num_features = num_features

        self.conv1 = [Conv2d(self.num_features, 128, 3, 2, 1)]
        self.conv1 += [Conv2d(128, 128, 3, 1, 1) for _ in range(3)]
        self.conv1 = nn.Sequential(*self.conv1)

        self.conv2 = [Conv2d(128, 128, 3, 2, 1)]
        self.conv2 += [Conv2d(128, 128, 3, 1, 1) for _ in range(5)]
        self.conv2 = nn.Sequential(*self.conv2)

        self.conv3 = [Conv2d(128, 256, 3, 2, 1)]
        self.conv3 += [Conv2d(256, 256, 3, 1, 1) for _ in range(5)]
        self.conv3 = nn.Sequential(*self.conv3)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 4, 0),nn.BatchNorm2d(256))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 256, 2, 2, 0),nn.BatchNorm2d(256))
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(128, 256, 1, 1, 0),nn.BatchNorm2d(256))

        self.score_head = Conv2d(768, self.num_classes, 1, 1, 0, activation=False, batch_norm=False)
        self.reg_head = Conv2d(768, self.num_regression_offsets * self.num_classes, 1, 1, 0, activation=False, batch_norm=False)
        self.dir_head = Conv2d(768, self.num_directions * self.num_classes, 1, 1, 0, activation=False, batch_norm=False)

    def forward(self, features: Tensor) -> Tuple[Tensor]:
        """
        Forward function.

        Args:
            features (Tensor): Convolutional features.

        Returns:
            Tuple[Tensor]: Score map for classification, Regression map for Bounding Boxes and Direction map.
        """
        features = self.conv1(features)
        x_skip_1 = features
        features = self.conv2(features)
        x_skip_2 = features
        features = self.conv3(features)
        
        x_0 = self.deconv1(features)
        x_1 = self.deconv2(x_skip_2)
        x_2 = self.deconv3(x_skip_1)
        
        x_2 = F.pad(x_2, (0,1), mode='constant', value=0)
        
        cat = torch.cat((x_0,x_1,x_2),1)
        
        score = self.score_head(cat)
        regression = self.reg_head(cat)
        direction = self.dir_head(cat)
        
        score = F.softmax(score, dim=-1)
        direction = F.sigmoid(direction)
        
        return score, regression, direction
