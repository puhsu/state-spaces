import json
import logging
from pathlib import Path
from typing import Dict

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def setup_logging():
    class TqdmLoggingHandler(logging.Handler):
        def __init__(self, level=logging.NOTSET):
            super().__init__(level)

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(TqdmLoggingHandler())


def log_metrics(writer: SummaryWriter, metrics: Dict[str, float], examples_seen, *, prefix: str = '') -> None:
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(f'{prefix}_{metric_name}', metric_value, global_step=examples_seen)


def dump_metrics(
        log_file: Path,
        run_args: Dict[str, str],
        val_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        target_metric: str
) -> None:
    with open(log_file, 'a') as f:
        f.write(f'{json.dumps(run_args)},{json.dumps(val_metrics)},{json.dumps(test_metrics)},'
                f'{val_metrics[target_metric]},{test_metrics[target_metric]}\n')
