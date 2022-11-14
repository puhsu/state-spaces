from ss_datasets.lra.loader import PathFinder


def configure_lra(x: int = 128, data_dir: str = 'ss_datasets/lra/lra_release/') -> PathFinder:
    """
    :param x: 64, 128 or 256
    :param data_dir: Path to downloaded dataset folder
    Usage example:
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
    pf = PathFinder(_name_='pathfinder', data_dir=Path(data_dir).join(Path(f'lra_release/pathfinder{x}/')))
    pf.setup()
    return pf
