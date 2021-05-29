import torch.nn as nn
import torch.nn.functional as F

from model import ResidualStack


class Encoder(nn.Module):
    def __init__(self, in_Channels, numHiddens, numResidualLayers, numResidualHiddens):
        super(Encoder, self).__init__()

        self._conv1 = nn.Conv2d(in_channels=in_Channels,
                                out_channels=numHiddens // 2,
                                kernel_size=4,
                                stride=2, padding=1)
        self._conv2 = nn.Conv2d(in_channels=numHiddens // 2,
                                out_channels=numHiddens,
                                kernel_size=4,
                                stride=2, padding=1)
        self._conv3 = nn.Conv2d(in_channels=numHiddens,
                                out_channels=numHiddens,
                                kernel_size=3,
                                stride=1, padding=1)
        self._residualStack = ResidualStack.ResidualStack(inChannels=numHiddens,
                                                          numHiddens=numHiddens,
                                                          numResidualLayers=numResidualLayers,
                                                          numResidualHiddens=numResidualHiddens)

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = F.relu(x)

        x = self._conv2(x)
        x = F.relu(x)

        x = self._conv3(x)
        return self._residualStack(x)
