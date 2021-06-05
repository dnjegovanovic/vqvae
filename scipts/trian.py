import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from six.moves import xrange

import umap

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.nn.functional as F

from model.Model import Model
from utils.utils import show

if torch.cuda.is_available():
    print(True)

def load_data():
    training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                     ]))

    validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                       ]))

    return training_data, validation_data


def train_vqvae():
    trainingData, validationData = load_data()
    dataVariance = np.var(trainingData.data / 255.0)
    print(dataVariance)

    batchSize = 256
    numTrainingUpdates = 15000

    numHiddens = 128
    numResidualHiddens = 32
    numResidualLayers = 2

    embeddingDim = 64
    numEmbeddings = 512

    commitmentCost = 0.25

    decay = 0.99

    learningRate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainingLoader = DataLoader(trainingData,
                                batch_size=batchSize,
                                shuffle=True,
                                pin_memory=True)

    validationLoader = DataLoader(validationData,
                                  batch_size=32,
                                  shuffle=True,
                                  pin_memory=True)

    model = Model(numHiddens, numResidualLayers, numResidualHiddens,
                  numEmbeddings, embeddingDim,
                  commitmentCost, decay).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learningRate, amsgrad=False)

    model.train()
    trainResReconError = []
    trainResPerplexity = []

    for i in xrange(numTrainingUpdates):
        (data, _) = next(iter(trainingLoader))
        data = data.to(device)
        optimizer.zero_grad()

        vqLoss, dataRecon, perplexity = model(data)
        reconError = F.mse_loss(dataRecon, data) / dataVariance
        loss = reconError + vqLoss
        loss.backward()

        optimizer.step()

        trainResReconError.append(reconError.item())
        trainResPerplexity.append(perplexity.item())

        if (i + 1) % 100 == 0:
            print('%d iterations' % (i + 1))
            print('recon_error: %.3f' % np.mean(trainResReconError[-100:]))
            print('perplexity: %.3f' % np.mean(trainResPerplexity[-100:]))
            print()

    torch.save(model.state_dict(), 'vqvae_model_weights2.pth')

    trainResReconErrorSmooth = savgol_filter(trainResReconError, 201, 7)
    trainResPerplexitySmooth = savgol_filter(trainResPerplexity, 201, 7)

    f = plt.figure(figsize=(16, 8))
    ax = f.add_subplot(1, 2, 1)
    ax.plot(trainResReconErrorSmooth)
    ax.set_yscale('log')
    ax.set_title('Smoothed NMSE.')
    ax.set_xlabel('iteration')

    ax = f.add_subplot(1, 2, 2)
    ax.plot(trainResPerplexitySmooth)
    ax.set_title('Smoothed Average codebook usage (perplexity).')
    ax.set_xlabel('iteration')

    plt.savefig('./imgs/train_graph2.png')

    model.eval()

    (validOriginals, _) = next(iter(validationLoader))
    validOriginals = validOriginals.to(device)

    vqOutputEval = model._preVqConv(model._encoder(validOriginals))
    _, validQuantize, _, _ = model._vqVae(vqOutputEval)
    validReconstructions = model._decoder(validQuantize)

    (trainOriginals, _) = next(iter(trainingLoader))
    trainOriginals = trainOriginals.to(device)
    _, trainReconstructions, _, _ = model._vqVae(trainOriginals)

    show(make_grid(validReconstructions.cpu().data) + 0.5, "validRec2")
    show(make_grid(validOriginals.cpu() + 0.5), "validOrg2")

    f2 = plt.figure(figsize=(16, 8))
    proj = umap.UMAP(n_neighbors=3,
                     min_dist=0.1,
                     metric='cosine').fit_transform(model._vqVae._embedding.weight.data.cpu())

    plt.scatter(proj[:, 0], proj[:, 1], alpha=0.3)
    plt.savefig('./imgs/embedding2.png')
