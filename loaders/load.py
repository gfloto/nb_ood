from einops import rearrange
from functools import partial
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

def fetch_loader(hps):
    assert hps.dataset in ['cifar10', 'cifar100']
    train = True if not hps.test else False

    if hps.dataset == 'cifar10':
        transforms_ = transforms.Compose([
            transforms.ToTensor(),
            lambda x: rearrange(x, 'c h w -> h w c'),
        ])

        partial_dataset = partial(
            datasets.CIFAR10, root='./data',
            train=train, transform=transforms_,
        )

        try: dataset = partial_dataset(download=False)
        except: dataset = partial_dataset(download=True)

        loader = DataLoader(
            dataset, batch_size=hps.batch_size,
            shuffle=True, num_workers=hps.num_workers,
        )

    elif hps.dataset == 'cifar100':
        transforms_ = transforms.Compose([
            transforms.ToTensor(),
            lambda x: rearrange(x, 'c h w -> h w c'),
        ])

        partial_dataset = partial(
            datasets.CIFAR100, root='./data',
            train=train, transform=transforms_,
        )

        try: dataset = partial_dataset(download=False)
        except: dataset = partial_dataset(download=True)

        loader = DataLoader(
            dataset, batch_size=hps.batch_size,
            shuffle=True, num_workers=hps.num_workers,
        )

    return loader