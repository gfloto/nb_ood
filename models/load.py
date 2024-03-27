import torch

def fetch_model(hps):
    assert hps.task in ['pca']

    if hps.task == 'pca':
        from models.siren import EigenFunction

        model = EigenFunction

    return model
    