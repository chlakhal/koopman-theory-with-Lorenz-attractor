# methods/simulate.py
import torch

def rollout_linear(A, x0, steps):
    traj = [x0]
    x = x0

    for _ in range(steps):
        x = A @ x
        traj.append(x)

    return torch.stack(traj)