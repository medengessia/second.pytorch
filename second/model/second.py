from typing import List
from torch import nn, Tensor
from .layers.vfe import VFExtractor
from .layers.sparse_conv import SparseCNN3D
from .layers.region_proposal import RPN


class SECOND(nn.Module):
    """
    Implements the state-of-the-art model SECOND.

    Args:        
        in_features (int): Input features.
        
        num_classes (int): The number of classes.
        
        num_directions (int): The number of directions.
        
        num_regression_offsets (int): The number of regression offsets.
        
        num_features_1 (int, optional): Number of features to use for the first VFE layer. 
        Defaults to 32.
        
        num_features_2 (int, optional): Number of features to use for the second VFE layer. 
        64 for a smaller network, 128 for a larger one. Defaults to 128.
        
        num_features_3 (int, optional): Number of features to use for the Linear layer. 
        It will then be used for the sparse convolutions and the RPN. Defaults to 128.
        
        shape (List[int], optional): the shape of the desired output for sparse convolution. 
        [128,10,320,264] for a smaller network, [128,10,400,350] for a larger one. Defaults to [128,10,400,350].
    """
    def __init__(self,
                in_features: int,
                num_classes: int,
                num_directions: int,
                num_regression_offsets: int,                
                num_features_1: int = 32,
                num_features_2: int = 128,
                num_features_3: int = 128,
                shape: List[int] = [128,10,400,350]) -> None:
        super(SECOND, self).__init__()
        
        self.shape = shape
        self.in_features = in_features
        
        self.num_features_1 = num_features_1
        self.num_features_2 = num_features_2
        self.num_features_3 = num_features_3
        
        self.num_classes = num_classes
        self.num_directions = num_directions
        self.num_regression_offsets = num_regression_offsets
        
        self.vf_extractor = VFExtractor(
                                        in_features=self.in_features,
                                        num_features_1=self.num_features_1,
                                        num_features_2=self.num_features_2,
                                        num_features_3=self.num_features_3
                                        )
        self.sparse_conv = SparseCNN3D(shape=self.shape)
        self.rpn = RPN( 
                    num_features=self.num_features_3,
                    num_classes=self.num_classes,
                    num_regression_offsets=self.num_regression_offsets,
                    num_directions=self.num_directions
                    )
        
    def forward(self, features: Tensor, coors: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            features (Tensor): Point features.
            coors (Tensor): Point coordinates.

        Returns:
            Tensor: Score map for classification, Regression map for Bounding Boxes and Direction map.
        """
        batch_size = features.shape[0]
        
        # Voxel Features Extractor
        out = self.vf_extractor(features) # (10 x 400 x 350) x 128 Tensor
        
        # Sparse 3D CNN
        out = self.sparse_conv(out, coors, batch_size) # 1 x 64 x 2 x 400 x 350 Tensor
        
        # Reshape features to fit RPN input format
        out = out.view(1, self.shape[0], self.shape[2], self.shape[3])
        
        # Region Proposal Network
        out = self.rpn(out)
        
        return out