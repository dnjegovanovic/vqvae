import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ResidualStack


class Decoder(nn.Module):
    def __init__(self, inChannels, numHiddens, numResidualLayers, numResidualHiddens):
        super(Decoder, self).__init__()

        self._conv1 = nn.Conv2d(in_channels=inChannels,
                                out_channels=numHiddens,
                                kernel_size=3,
                                stride=1, padding=1)

        self._residualStack = ResidualStack.ResidualStack(inChannels=numHiddens,
                                                          numHiddens=numHiddens,
                                                          numResidualLayers=numResidualLayers,
                                                          numResidualHiddens=numResidualHiddens)

        self._convTrans1 = nn.ConvTranspose2d(in_channels=numHiddens,
                                              out_channels=numHiddens // 2,
                                              kernel_size=4,
                                              stride=2, padding=1)

        self._convTrans2 = nn.ConvTranspose2d(in_channels=numHiddens // 2,
                                              out_channels=3,
                                              kernel_size=4,
                                              stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv1(inputs)

        x = self._residualStack(x)

        x = self._convTrans1(x)
        x = F.relu(x)

        return self._convTrans2(x)


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder(40, 128, 3, 64)
    decoder_out = decoder(x)
    print('Dncoder out shape:', decoder_out.shape)
