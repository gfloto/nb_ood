import math
import torch
from einops import rearrange
import matplotlib.pyplot as plt

def plot_recon(orig, recon, save_path):
    orig = torch.clamp(orig, 0, 1)
    recon = torch.clamp(recon, 0, 1)

    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.imshow(orig)
    ax1.set_title('original')
    ax1.axis('off')

    ax2.imshow(recon)
    ax2.set_title('reconstructed')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_basis(basis, eigen_f, save_path):
    x = torch.cat([basis, eigen_f[None, ...]], dim=0)
    nb, h, w, c = x.shape

    # normalize
    x = (x - x.mean()) / x.std()
    x = torch.sigmoid(x)

    # pad images with zeros
    p = 1
    zeros = torch.zeros(nb, h+2*p, w+2*p, c).cuda()
    zeros[:, p:-p, p:-p, :] = x
    x = zeros

    # reshape to square with tiles
    # first add empty tiles
    k = math.ceil(nb ** 0.5)
    add_tiles = torch.zeros(k**2 - nb, h+2*p, w+2*p, c).cuda()
    x = torch.cat([x, add_tiles], dim=0)

    x = rearrange(x, '(k1 k2) h w c -> (k1 h) (k2 w) c', k1=k, k2=k).cpu()

    
    # plot
    fig = plt.figure(figsize=(10, 10))

    plt.imshow(x)
    plt.title('basis functions')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()