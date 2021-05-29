import torch.nn as nn


class ResidualLayer(nn.Module):
    def __init__(self, inChannels, numHiddens, numResidualHiddens):
        super(ResidualLayer, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=inChannels,
                      out_channels=numResidualHiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=numResidualHiddens,
                      out_channels=numHiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)
