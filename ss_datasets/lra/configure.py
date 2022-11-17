from pathlib import Path

from ss_datasets.lra.loader import PathFinder


def configure_lra(x: int = 128, tokenize: bool = False, data_dir: str = 'ss_datasets/lra/lra_release/') -> PathFinder:
    """
    :param x: 64, 128 or 256
    :param tokenize: convert to 0-255 range
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
    pf = PathFinder(_name_='pathfinder', tokenize=tokenize, data_dir=Path(data_dir).joinpath(Path(f'lra_release/pathfinder{x}/')))
    pf.setup()
    return pf
