
def fetch_train_eval(hps):
    assert hps.task in ['pca']

    if hps.task == 'pca':
        from train_eval.pca import train_eval_pca
        return train_eval_pca