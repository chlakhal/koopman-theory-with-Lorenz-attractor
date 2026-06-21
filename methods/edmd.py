# methods/edmd.py
import torch

def phi(x):
    x1, x2, x3 = x

    return torch.tensor([
        x1, x2, x3,
        x1*x1, x1*x2, x1*x3,
        x2*x2, x2*x3,
        x3*x3
    ], dtype=torch.float32)


def build_lifted_data(data):
    Phi_X = torch.stack([phi(x) for x in data[:-1]]).T
    Phi_Y = torch.stack([phi(x) for x in data[1:]]).T

    return Phi_X, Phi_Y


def edmd(Phi_X, Phi_Y):
    U, S, Vh = torch.linalg.svd(Phi_X, full_matrices=False)

    S_inv = torch.diag(1 / S)

    K = Phi_Y @ Vh.T @ S_inv @ U.T

    return K