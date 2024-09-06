import torch
import spconv.pytorch as spconv
from typing import List
from torch import nn, Tensor
from einops import rearrange


class SparseCNN3D(nn.Module):
    """
    Implements the Sparse Convolutional Layers used in SECOND.

    Args:
        shape (List[int]): the shape of the desired output.
    """
    def __init__(self, shape: List[int]) -> None:
        super(SparseCNN3D, self).__init__()
        self.shape = shape
        
        self.sparse_cnn = spconv.SparseSequential(
        
            spconv.SubMConv3d(
                self.shape[0], 64,
                kernel_size=(3, 1, 1),
                stride=(2, 1, 1),
                algo=spconv.ConvAlgo.Native,
                indice_key="subm1"
            ),
            nn.LeakyReLU(),
            
            spconv.SparseConv3d(
                64, 64,
                kernel_size=(3, 1, 1),
                stride=(2, 1, 1),
                algo=spconv.ConvAlgo.Native,
                indice_key="spconv1"
            ),
            nn.LeakyReLU(),
            
            spconv.SubMConv3d(
                64, 64,
                kernel_size=(3, 1, 1),
                stride=(2, 1, 1),
                algo=spconv.ConvAlgo.Native,
                indice_key="subm2"
            ),
            nn.LeakyReLU(),
            
            spconv.SubMConv3d(
                64, 64,
                kernel_size=(3, 1, 1),
                stride=(2, 1, 1),
                algo=spconv.ConvAlgo.Native,
                indice_key="subm3"
            ),
            nn.LeakyReLU(),
            
            spconv.SparseConv3d(
                64, 64,
                kernel_size=(3, 1, 1),
                stride=(2, 1, 1),
                algo=spconv.ConvAlgo.Native,
                indice_key="spconv2"
            ),
            nn.LeakyReLU(),
            
            spconv.ToDense()
        
        )
        
    def forward(self, features: Tensor, coors: Tensor, batch_size: int) -> Tensor:
        """
        Forward function.

        Args:
            features (Tensor): Point features.
            coors (Tensor): Point coordinates.
            batch_size (int): Batch size.
        
        Returns:
            Tensor: Extracted features from sparse convolutions.
        """
        # Combine batch and voxel indices for SparseConvTensor
        coors = rearrange(coors, 'b n d -> (b n) d')
        coors = coors.int()
        
        # Give an odd number to the first shape 10, like in the paper
        shape = torch.tensor(self.shape[1:])
        shape_bias = torch.zeros_like(shape)
        shape_bias[0][shape[0] % 2 == 0] = 1
        shape += shape_bias
        
        out = spconv.SparseConvTensor(features, coors, shape, batch_size)
        out = self.sparse_cnn(out)
        
        return out