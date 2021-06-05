import numpy as np
import torch
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from model.Model import Model
from utils.utils import show

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generateSamples(model, e_indices, numEmbeddings, batchSize, embeddingDim):
    minEncodings = torch.zeros(e_indices.shape[0], numEmbeddings).to(device)
    minEncodings.scatter_(1, e_indices, 1)
    e_weights = model._vqVae._embedding.weight
    z_q = torch.matmul(minEncodings, e_weights).view((batchSize, 8, 8, embeddingDim))
    z_q = z_q.permute(0, 3, 1, 2).contiguous()

    x_recon = model._decoder(z_q)
    return x_recon, z_q, e_indices


def uniformSamples(model, numEmbeddings, batchSize, embeddingDim):
    rand = np.random.randint(numEmbeddings, size=(2048, 1))
    minEncodingIndices = torch.tensor(rand).long().to(device)
    xRecon, z_q, eIndices = generateSamples(model, minEncodingIndices,
                                             numEmbeddings, batchSize, embeddingDim)

    print(minEncodingIndices.shape)
    return xRecon, z_q, eIndices


def generateUniform():
    numHiddens = 128
    numResidualHiddens = 32
    numResidualLayers = 2

    embeddingDim = 64
    numEmbeddings = 512

    commitmentCost = 0.25
    batchSize = 32

    decay = 0.99
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(numHiddens, numResidualLayers, numResidualHiddens,
                  numEmbeddings, embeddingDim,
                  commitmentCost, decay).to(device)

    model.load_state_dict(torch.load('../vqvae_model_weights.pth'))
    model.eval()

    x_val_recon, z_q, e_indices = uniformSamples(model, numEmbeddings, batchSize, embeddingDim)

    show(make_grid(x_val_recon.cpu().data) + 0.5, "../generate_uniform")




if __name__ == '__main__':
    generateUniform()
