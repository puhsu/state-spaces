from ss_datasets.lra.loader import PathFinder


def configure_lra() -> PathFinder:
    pf = PathFinder('lra_release/lra_release/pathfinder128/')
    pf.setup()
    return pf
