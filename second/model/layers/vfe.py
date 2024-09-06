import torch
from torch import nn, Tensor
from einops import rearrange


# Fully Connected Network
class FCN(nn.Module):
    """
    Fully connected network of SECOND.

    Args:
        in_features (int): Number of input features.
        num_features (int): Number of features to extract.
    """
    def __init__(self, in_features: int, num_features: int) -> None:
        super(FCN, self).__init__()
        self.in_features = in_features
        self.num_features = num_features
        
        self.fcn = nn.Sequential(
            nn.Linear(self.in_features, self.num_features),
            nn.BatchNorm1d(self.num_features),
            nn.LeakyReLU()
        )

    def forward(self, features: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            features (Tensor): Point features.

        Returns:
            Tensor: the output linear features extracted from the voxels.
        """
        batch_size, num_voxels, max_points, in_features = features.shape
        out = features.view(-1, in_features)
        out = self.fcn(out)
        out = rearrange(out, '(b n m) f -> b n m f', b=batch_size, n=num_voxels, m=max_points)
        return out

# Voxel Feature Encoding layer
class VFE(nn.Module):
    """
    Voxel feature encoder used in SECOND.

    It applies the layers described in the Paper, i.e. a Linear, 
    a BatchNorm and a ReLU for the FCN part, and a MaxPooling layer.

    Args:
        in_features (int): Number of input features.
        num_features (int): Number of features to extract.
    """
    def __init__(self, in_features: int, num_features: int) -> None:
        super(VFE, self).__init__()
        assert num_features % 2 == 0
        
        self.units = num_features // 2
        self.fcn = FCN(in_features,self.units)

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            features (Tensor): Point features.
            mask (Tensor): A mask to apply to the point-wise features.

        Returns:
            Tensor: the locally aggregated features for each voxel.
        """
        _, _, max_points, _ = features.shape
        
        # point-wise features
        pwf = self.fcn(features)
        
        # Locally aggregated feature
        laf = torch.max(pwf, 2)[0].unsqueeze(2).repeat(1, 1, max_points, 1)
        
        # Point-wise concat feature
        pwcf = torch.cat((pwf, laf),dim=3)
        
        # Apply mask
        mask = mask.unsqueeze(3).repeat(1, 1, 1, self.units * 2)
        pwcf = pwcf * mask.float()

        return pwcf
    
# Stacked Voxel Feature Encoding
class VFExtractor(nn.Module):
    """
    Voxel Feature Extractor used in SECOND.
    
    It applies the layers described in the Paper, i.e. two VFEs and a Linear layer.

    Args:
        in_features (int): Number of input features.
        
        num_features_1 (int): Number of features to extract for the first VFE layer.
        
        num_features_2 (int): Number of features to extract for the second VFE layer.
        
        num_features_3 (int): Number of features to extract for the Linear layer.
    """
    def __init__(self,
                in_features: int,
                num_features_1: int,
                num_features_2: int,
                num_features_3: int) -> None:
        super(VFExtractor, self).__init__()
        self.in_features = in_features
        self.num_features_1 = num_features_1
        self.num_features_2 = num_features_2
        self.num_features_3 = num_features_3
        
        self.vfe_1 = VFE(self.in_features, self.num_features_1)
        self.vfe_2 = VFE(self.num_features_1, self.num_features_2)
        self.fcn = FCN(num_features_2, num_features_3)
        
    def forward(self, features: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            features (Tensor): Point features.

        Returns:
            Tensor: the output sparse features extracted from the voxels.
        """
        mask = torch.ne(torch.max(features, 3)[0], 0)
        out = self.vfe_1(features, mask)
        out = self.vfe_2(out, mask)
        out = self.fcn(out)
        out = out.view(-1, self.num_features_3)
        return out
