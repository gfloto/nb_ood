import os
import json
import argparse
import numpy as np
import torch

from loaders.load import fetch_loader 
from train_eval.load import fetch_train_eval
from models.load import fetch_model

def get_hps():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_path', type=str, default='dev')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--task', type=str, default='pca')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)

    # neural-basis params
    parser.add_argument('--n_basis', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--dim_hidden', type=int, default=128)
    parser.add_argument('--eigen_reg', type=float, default=1.0)

    hps = parser.parse_args()
    
    assert hps.task in ['pca']
    assert hps.dataset in ['cifar10', 'cifar100']

    # set number of channels based of dataset
    if hps.dataset in ['cifar10', 'cifar100']:
        hps.channels = 3
        hps.height = 32; hps.width = 32

    # save hps to .json
    save_hps(hps)

    return hps

# save hps to .json
def save_hps(hps):
    if not os.path.exists('results'): os.makedirs('results')
    hps.exp_path = os.path.join('results', hps.exp_path)
    if not os.path.exists(hps.exp_path): os.makedirs(hps.exp_path)

    save_path = os.path.join(hps.exp_path, 'hps.json')
    with open(save_path, 'w') as f:
        json.dump(hps.__dict__, f, indent=2)

if __name__ == '__main__':
    hps = get_hps()

    loader = fetch_loader(hps)
    Model = fetch_model(hps)
    train_eval = fetch_train_eval(hps)

    # if test, don't train
    if hps.test: mode = 'test'
    else: mode = 'train'

    train_eval(Model, loader, mode, hps)
