# train/train_deep_koopman.py
import torch
import torch.nn as nn
from utils.plot import plot_loss

def train(model, data, epochs=200, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # X : états de t=0 à N-2
    # X_next : états de t=1 à N-1
    X = torch.tensor(data[:-1], dtype=torch.float32)
    X_next = torch.tensor(data[1:], dtype=torch.float32)

    losses = []

    for epoch in range(epochs):
        opt.zero_grad()

        # Forward pass
        x_rec, z, z_next_pred = model(X)
        z_next_true = model.ae.encode(X_next)

        # Alignement temporel : z_t+1 (true) vs z_t (pred)
        # Les deux ont longueur N-1
        loss_rec = loss_fn(x_rec, X)
        loss_dyn = loss_fn(z_next_pred, z_next_true)

        loss = loss_rec + loss_dyn

        loss.backward()
        opt.step()

        losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: {loss.item():.4f}")

    # Sauvegarde de la courbe de loss
    plot_loss(losses, filename="deepkoopman_loss_2.png")
