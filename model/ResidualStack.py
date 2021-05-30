import numpy as np
import torch
from model.Residual import ResidualLayer
import torch.nn as nn
import torch.nn.functional as F


class ResidualStack(nn.Module):
    def __init__(self, inChannels, numHiddens, numResidualLayers, numResidualHiddens):
        super(ResidualStack, self).__init__()
        self._numResiduaLayers = numResidualLayers
        self._layers = nn.ModuleList([ResidualLayer(inChannels, numHiddens, numResidualHiddens)
                                      for _ in range(self._numResiduaLayers)])

    def forward(self, x):
        for i in range(self._numResiduaLayers):
            x = self._layers[i](x)
        return F.relu(x)


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()
    # test res stack
    res_stack = ResidualStack(40, 40, 20, 3)
    res_stack_out = res_stack(x)
    print('Res Stack out shape:', res_stack_out.shape)
