# data/lorenz.py
import numpy as np
from scipy.integrate import solve_ivp

def lorenz(t, state, sigma=10, beta=8/3, rho=28):
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]


def generate_lorenz(T=30, steps=3000):
    t = np.linspace(0, T, steps)
    x0 = [1.0, 1.0, 1.0]

    sol = solve_ivp(lorenz, (0, T), x0, t_eval=t)

    return sol.y.T  # (N, 3)