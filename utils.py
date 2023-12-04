# Helper function for visualization.
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize(h, color, file_name: str ='visualization.png'):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.savefig(file_name)
    plt.close()

