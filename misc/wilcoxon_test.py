import numpy as np

from scipy.stats import wilcoxon

def test_samples(x: np.ndarray, y: np.ndarray) -> float:
    if (x == y).all():
        return 1.
    result = wilcoxon(x=x, y=y, alternative="two-sided")
    return result.pvalue
