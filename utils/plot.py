# utils/plot.py
import matplotlib.pyplot as plt

def plot_3d(true, pred=None, filename="trajectory.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(true[:,0], true[:,1], true[:,2], label="True", lw=0.5)

    if pred is not None:
        ax.plot(pred[:,0], pred[:,1], pred[:,2], label="Pred", lw=0.5)

    ax.legend()
    ax.set_title("3D Trajectory")

    # Sauvegarde au lieu d'afficher
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def plot_loss(losses, filename="loss_curve.png"):
    plt.figure()
    plt.plot(losses, lw=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close()
