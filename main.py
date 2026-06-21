# main.py
import torch
import matplotlib.pyplot as plt

from data.lorenz import generate_lorenz
from methods.dmd import dmd
from methods.simulate import rollout_linear
from methods.edmd import build_lifted_data, edmd, phi
from models.deep_koopman import DeepKoopman
from train.train_deep_koopman import train
from utils.plot import plot_3d

# -----------------------
# 1. Trajectoire vraie
# -----------------------
data = generate_lorenz()
plot_3d(torch.tensor(data), filename="lorenz_true_2.png")

# -----------------------
# 2. DMD baseline
# -----------------------
X = torch.tensor(data[:-1].T, dtype=torch.float32)
Y = torch.tensor(data[1:].T, dtype=torch.float32)
A = dmd(X, Y)

x0 = torch.tensor(data[0], dtype=torch.float32)
dmd_traj = rollout_linear(A, x0, len(data)-1)
plot_3d(torch.tensor(data), dmd_traj, filename="lorenz_dmd_2.png")

# -----------------------
# 3. EDMD
# -----------------------
Phi_X, Phi_Y = build_lifted_data(data)
K = edmd(Phi_X, Phi_Y)

traj_edmd = rollout_linear(K, phi(torch.tensor(data[0], dtype=torch.float32)), len(data)-1)
plot_3d(torch.tensor(data), traj_edmd.T, filename="lorenz_edmd_2.png")

# -----------------------
# 4. Deep Koopman
# -----------------------
model = DeepKoopman()
train(model, data)

# Pour les métriques, on utilise X et X_next
X_full = torch.tensor(data[:-1], dtype=torch.float32)
X_next_full = torch.tensor(data[1:], dtype=torch.float32)

x_rec, z, z_next_pred = model(X_full)
z_next_true = model.ae.encode(X_next_full)

plot_3d(X_full, x_rec.detach(), filename="lorenz_deepkoopman_2.png")

# -----------------------
# 5. Métriques
# -----------------------
loss_rec = torch.mean((X_full - x_rec)**2).item()
loss_dyn = torch.mean((z_next_true - z_next_pred)**2).item()

print("Reconstruction Loss:", loss_rec)
print("Dynamics Loss:", loss_dyn)

# -----------------------
# 6. Comparaison directe (subplot)
# -----------------------
fig = plt.figure(figsize=(12, 10))

ax1 = fig.add_subplot(221, projection='3d')
ax1.plot(data[:,0], data[:,1], data[:,2], lw=0.5, label="True")
ax1.set_title("Lorenz True")
ax1.legend()

ax2 = fig.add_subplot(222, projection='3d')
ax2.plot(data[:,0], data[:,1], data[:,2], lw=0.5, label="True")
ax2.plot(dmd_traj[:,0], dmd_traj[:,1], dmd_traj[:,2], lw=0.5, label="DMD")
ax2.set_title("DMD Approximation")
ax2.legend()

ax3 = fig.add_subplot(223, projection='3d')
ax3.plot(data[:,0], data[:,1], data[:,2], lw=0.5, label="True")
ax3.plot(traj_edmd[0,:], traj_edmd[1,:], traj_edmd[2,:], lw=0.5, label="EDMD")
ax3.set_title("EDMD Approximation")
ax3.legend()

ax4 = fig.add_subplot(224, projection='3d')
ax4.plot(data[:,0], data[:,1], data[:,2], lw=0.5, label="True")
ax4.plot(x_rec.detach()[:,0], x_rec.detach()[:,1], x_rec.detach()[:,2], lw=0.5, label="DeepKoopman")
ax4.set_title("DeepKoopman Reconstruction")
ax4.legend()

plt.tight_layout()
plt.savefig("comparison_all_2.png", dpi=300)
plt.close(fig)
