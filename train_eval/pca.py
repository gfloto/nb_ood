import os
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from einops import rearrange, repeat

from plotting import plot_recon, plot_basis

def func_inner(x, y):
    assert x.shape == y.shape
    return torch.einsum('... h w c, ... h w c -> ... c', x, y)

def func_norm(x):
    return func_inner(x, x).sqrt()

def make_domain(h, w):
    xh = torch.linspace(0, 1, h).cuda()
    xw = torch.linspace(0, 1, w).cuda()
    xh, xw = torch.meshgrid(xh, xw, indexing='ij')
    dom = torch.stack([xh, xw], dim=-1) 

    return dom

# take residual of x by removing ontributions from last learned basis components
def basis_residual(x, basis):
    coeffs = torch.einsum('b h w c, e h w c -> b e c', x, basis)
    recon = torch.einsum('b e c, e h w c -> b h w c', coeffs, basis)

    return x - recon

# ensure that new eigen-function is orthogonal to previous basis
def orthogonalize(eigen_f, basis):
    coeff = torch.einsum('h w c, e h w c -> e c', eigen_f, basis)
    recon = torch.einsum('e c, e h w c -> h w c', coeff, basis)
    eigen_f -= recon

    return eigen_f

# reconstruct x using basis and new eigen-function
def reconstruct(x, basis, eigen_f):
    full_basis = torch.cat([basis, eigen_f[None, ...]], dim=0)
    coeffs = torch.einsum('b h w c, e h w c -> b e c', x, full_basis)
    recon = torch.einsum('b e c, e h w c -> b h w c', coeffs, full_basis)

    return recon

def save_basis(eigen_f, basis, save_path):
    new_basis = torch.cat([ basis, eigen_f[None, ...] ], dim=0)
    torch.save(new_basis, save_path)

    return new_basis

def train(model, loader, optim, basis, hps): 
    model.train()

    h, w, c = hps.height, hps.width, hps.channels

    # domain of eigen-function
    domain = make_domain(hps.height, hps.width).cuda()

    for _ in range(25):
        loss_track = []
        for i, (x, _) in enumerate(tqdm(loader)):
            # train on the residual
            x = x.cuda()
            x_res = basis_residual(x, basis)

            # get eigen-function, stack copies into batch
            eigen_f = model(domain)

            # ensure that eigen-function is normalized and orthogonal to basis
            eigen_f = orthogonalize(eigen_f, basis)
            eigen_f = eigen_f / func_norm(eigen_f)[None, None, :]

            # different losses
            inner = func_inner(
                repeat(eigen_f, 'h w c -> b h w c', b=x_res.shape[0]),
                x_res,
            )
            
            loss = -inner.abs().mean()

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            # save losses
            loss_track.append(loss.item())

            # recon
            recon = reconstruct(x, basis, eigen_f)

            # plot basis
        print(f'loss: {np.mean(loss_track)}')

        plot_recon(
            x[0].detach().cpu(),
            recon[0].detach().cpu(),
            os.path.join(hps.exp_path, f'recon.png'),
        )

        plot_basis(
            basis.detach().cpu(),
            eigen_f.detach().cpu(),
            os.path.join(hps.exp_path, f'basis.png'),
        )

    print('saving basis')
    basis = save_basis(
        eigen_f.detach().cpu(),
        basis.detach().cpu(),
        os.path.join(hps.exp_path, 'basis.pt',)
    )

    return basis.cuda()

def test(model, loader, basis, hps):
    pass

def load_basis(hps):
    # load basis if exists
    if os.path.exists(f'{hps.exp_path}/basis.pt'):
        print('loading basis')
        basis = torch.load(f'{hps.exp_path}/basis.pt').cuda()
    else:
        # set first basis function to be constant
        constant = torch.ones((
            1,
            hps.height,
            hps.width,
            hps.channels,
        )).cuda()

        # require that norm of basis function is 1
        norm = func_norm(constant)
        basis = constant / func_norm(constant)[:, None, None, :]

    return basis

def train_eval_pca(Model, loader, mode, hps):
    basis = load_basis(hps)

    if mode == 'train':
        for nb in range(basis.shape[0], hps.n_basis):
            print(f'training basis {nb}')

            # use new model for each basis
            model = Model(
                dim_in = 2, # x and y coordinates
                dim_hidden = hps.dim_hidden,
                dim_out = hps.channels,
                num_layers = hps.n_layers,
                w0 = 1.,
                w0_initial = 30.,
            ).cuda()

            optim = Adam(model.parameters(), lr=hps.lr)

            # train next principal function
            basis = train(model, loader, optim, basis, hps)

    elif mode == 'test':
        test(model, loader, basis, hps)
    