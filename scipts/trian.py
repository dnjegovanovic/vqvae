import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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

    training_data, validation_data = load_data()
    data_variance = np.var(training_data.data / 255.0)

    print(data_variance)
