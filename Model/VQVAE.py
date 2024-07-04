import torch
import torch.nn as nn
import numpy as np
from .encoder import Encoder
from .quantizer import VectorQuantizer
from .decoder import Decoder
from os import path


class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, output_dim, n_res_layers, n_embeddings, embedding_dim,
                 code_dim, beta, save_img_embedding_map=False, vit=None):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, code_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim, output_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

        self.vit = vit

    def forward(self, x, verbose=False):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, _, min_indices, predicted_min_indices, OR_predicted_min_indices, OR_portion, attention_weights = self.vector_quantization(z_e, self.vit)

        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity, z_e, min_indices, predicted_min_indices, OR_predicted_min_indices, OR_portion, attention_weights
    
    def load_weights(self, model_path):
        assert path.exists(model_path)
        weights = torch.load(model_path)
        self.load_state_dict(weights, strict=False)