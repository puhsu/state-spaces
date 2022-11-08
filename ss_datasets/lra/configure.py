from ss_datasets.lra.loader import PathFinder


def configure_lra() -> PathFinder:
    pf = PathFinder(_name_='pathfinder', data_dir='lra_release/lra_release/pathfinder128/')
    pf.setup()
    return pf
