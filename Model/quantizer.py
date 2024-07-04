import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, c_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.c_dim = c_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.c_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z, vit=None):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        if vit is not None:
            predicted_min_indices, OR_predicted_min_indices, OR_portion, attention_weights = vit(z)

            z = z.permute(0, 2, 3, 1).contiguous()

            min_encoding_indices = predicted_min_indices.argmax(dim=-1).squeeze()
            min_encoding_indices = min_encoding_indices.view(-1)
            min_encoding_indices = min_encoding_indices.unsqueeze(1)
            min_encodings = torch.zeros(
                min_encoding_indices.shape[0], self.n_e).to(device)
            min_encodings.scatter_(1, min_encoding_indices, 1)
            # get quantized latent vectors
            z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
            # z_q = z + (z_q - z).detach()
            
            if OR_predicted_min_indices is None:
                Z_q = z_q
            else:
                OR_min_encoding_indices = OR_predicted_min_indices.argmax(dim=-1).squeeze()
                OR_min_encoding_indices = OR_min_encoding_indices.view(-1)
                OR_min_encoding_indices = OR_min_encoding_indices.unsqueeze(1)
                OR_min_encodings = torch.zeros(
                    OR_min_encoding_indices.shape[0], self.n_e).to(device)
                OR_min_encodings.scatter_(1, OR_min_encoding_indices, 1)
                # get quantized latent vectors
                OR_z_q = torch.matmul(OR_min_encodings, self.embedding.weight).view(z.shape)
                # OR_z_q = z + (OR_z_q - z).detach()

                # compute the weighted sum of two quantized latent vectors
                OR_portion = OR_portion.view(z.shape[0], z.shape[1], z.shape[2], 1)
                attention_weights = attention_weights.view(z.shape[0], z.shape[1], z.shape[2], 1)
                Z_q = OR_portion * OR_z_q + (1 - OR_portion) * z_q

                # preserve gradients
                # Z_q = z + (Z_q - z).detach()

            Z_q = Z_q.permute(0, 3, 1, 2).contiguous()
            return None, Z_q, None, min_encodings, min_encoding_indices, predicted_min_indices, OR_predicted_min_indices, OR_portion, attention_weights
        else:
            # reshape z -> (batch, height, width, channel) and flatten
            z = z.permute(0, 2, 3, 1).contiguous()
            z_flattened = z.view(-1, self.c_dim)
            # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1) - 2 * \
                torch.matmul(z_flattened, self.embedding.weight.t())

            # find closest encodings
            min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
            min_encodings = torch.zeros(
                min_encoding_indices.shape[0], self.n_e).to(device)
            min_encodings.scatter_(1, min_encoding_indices, 1)

            # get quantized latent vectors
            z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

            # compute loss for embedding
            loss = torch.mean((z_q.detach() - z)**2) + self.beta * \
                torch.mean((z_q - z.detach()) ** 2)

            # preserve gradients
            z_q = z + (z_q - z).detach()

            # perplexity
            e_mean = torch.mean(min_encodings, dim=0)
            perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

            return loss, z_q, perplexity, min_encodings, min_encoding_indices, None, None, None, None
    
    def get_latent(self, min_encoding_indices, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        min_encoding_indices = min_encoding_indices.view(-1)
        min_encoding_indices = min_encoding_indices.unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q