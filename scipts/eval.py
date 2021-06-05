import torch
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from model.Model import Model
from utils.utils import show

if torch.cuda.is_available():
    print(True)


def eval():
    validationData = datasets.CIFAR10(root="data", train=False, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                      ]))

    validationLoader = DataLoader(validationData,
                                  batch_size=32,
                                  shuffle=True,
                                  pin_memory=True)
    numHiddens = 128
    numResidualHiddens = 32
    numResidualLayers = 2

    embeddingDim = 64
    numEmbeddings = 512

    commitmentCost = 0.25

    decay = 0.99
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(numHiddens, numResidualLayers, numResidualHiddens,
                  numEmbeddings, embeddingDim,
                  commitmentCost, decay).to(device)

    model.load_state_dict(torch.load('./vqvae_model_weights.pth'))
    model.eval()

    (validOriginals, _) = next(iter(validationLoader))
    validOriginals = validOriginals.to(device)

    vqOutputEval = model._preVqConv(model._encoder(validOriginals))
    _, validQuantize, _, _ = model._vqVae(vqOutputEval)
    validReconstructions = model._decoder(validQuantize)

    show(make_grid(validReconstructions.cpu().data) + 0.5, "eval_rec2")
