from model import Residual
import torch.nn as nn
import torch.nn.functional as F


class ResidualStack(nn.Module):
    def __init__(self, inChannels, numHiddens, numResidualLayers, numResidualHiddens):
        super(ResidualStack, self).__init__()
        self._numResiduaLayers = numResidualLayers
        self._layers = nn.ModuleList([Residual(inChannels, numHiddens, numResidualHiddens)
                                      for _ in range(self._numResiduaLayers)])

    def forward(self, x):
        for i in range(self._numResiduaLayers):
            x = self._layers[i](x)
        return F.relu(x)
