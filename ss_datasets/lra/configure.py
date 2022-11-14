from ss_datasets.lra.loader import PathFinder


def configure_lra(data_dir='ss_datasets/lra/lra_release/lra_release/pathfinder128/') -> PathFinder:
    """Usage example:
    >>> from ss_datasets.lra.configure import configure_lra
    >>> pf = configure_lra()
    >>> iterator = iter(pf.train_dataloader())
    >>> next(iterator)
    (tensor([[[-1.],
         [-1.],
         [-1.],
         ...,
         [-1.],
         [-1.],
         [-1.]]]), tensor([0]), {'rate': 1})
    """
    pf = PathFinder(_name_='pathfinder', data_dir=data_dir)
    pf.setup()
    return pf
