# models/deep_koopman.py
import torch.nn as nn
from models.autoencoder import Autoencoder
from models.koopman import KoopmanOperator

class DeepKoopman(nn.Module):

    def __init__(self, input_dim=3, latent_dim=6):
        super().__init__()

        self.ae = Autoencoder(input_dim, latent_dim)
        self.koopman = KoopmanOperator(latent_dim)

    def forward(self, x):
        z = self.ae.encode(x)
        z_next = self.koopman(z)
        x_rec = self.ae.decode(z)

        return x_rec, z, z_next