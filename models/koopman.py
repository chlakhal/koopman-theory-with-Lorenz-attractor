# models/koopman.py
import torch.nn as nn

class KoopmanOperator(nn.Module):

    def __init__(self, latent_dim=6):
        super().__init__()

        self.K = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, z):
        return self.K(z)