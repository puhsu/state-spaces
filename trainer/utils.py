from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score


def compute_metrics(predictions: np.ndarray, gold_labels: np.ndarray) -> Dict[str, float]:
    return {f'accuracy': accuracy_score(gold_labels, predictions)}
