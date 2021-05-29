import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerLayer(nn.Module):

    def __init__(self, numEmbeddings, embeddingDim, commitmentCost):
        super(VectorQuantizerLayer, set).__init__()

        self._embeddingDim = embeddingDim
        self._numEmbeddings = numEmbeddings
        self._commitmentCost = commitmentCost

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        inputShape = inputs.shape

        # Flatten input
        flatInput = inputs.view(-1, self._embeddingDim)

        # Calculate distances
        distances = (torch.sum(flatInput ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flatInput, self._embedding.weight.t()))

        # Encoding
        encodingIndices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encodingIndices.shape[0], self._numEmbeddings, device=inputs.device)
        encodings.scatter_(1, encodingIndices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(inputShape)

        # Loss
        eLatentLoss = F.mse_loss(quantized.detach(), inputs)
        qLatentLoss = F.mse_loss(quantized, inputs.detach())
        loss = qLatentLoss + self._commitmentCost * eLatentLoss

        quantized = inputs + (quantized - inputs).detach()
        avgProbs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avgProbs * torch.log(avgProbs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
