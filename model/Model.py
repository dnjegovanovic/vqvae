import torch.nn as nn

from model.Encoder import Encoder
from model.Decoder import Decoder
from model.VectorQuantizer import VectorQuantizerLayer
from model.VectorQuantizerEMA import VectorQuantizerLayerEMA


class Model(nn.Module):
    def __init__(self, numHiddens, numResidualLayers, numResidualHiddens,
                 numEmbeddings, embeddingDim, commitmentCost, decay=0):
        super(Model, self).__init__()

        self._encoder = Encoder(3, numHiddens,
                                numResidualLayers,
                                numResidualHiddens)
        self._preVqConv = nn.Conv2d(in_channels=numHiddens,
                                      out_channels=embeddingDim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vqVae = VectorQuantizerLayerEMA(numEmbeddings, embeddingDim,
                                                   commitmentCost, decay)
        else:
            self._vqVae = VectorQuantizerLayer(numEmbeddings, embeddingDim,
                                                commitmentCost)
        self._decoder = Decoder(embeddingDim,
                                numHiddens,
                                numResidualLayers,
                                numResidualHiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._preVqConv(z)
        loss, quantized, perplexity, _ = self._vqVae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity
