# methods/dmd.py
import torch

def dmd(X, Y):
    """
    X, Y: (state_dim, time)
    """

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)

    S_inv = torch.diag(1 / S)

    A = Y @ Vh.T @ S_inv @ U.T

    return A