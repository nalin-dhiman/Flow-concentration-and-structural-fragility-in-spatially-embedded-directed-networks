import pytest
import numpy as np
from src.pareto.analysis import compute_pareto_frontier, signed_dist, min_euclidean_dist

def test_pareto_frontier():
    # Simple case
    # Points: (1,10), (2,9), (3,8) -> All pareto
    # (2, 11) -> Dominated by (1,10) and (2,9)
    # (0.5, 12) -> Pareto
    
    points = np.array([
        [1.0, 10.0],
        [2.0, 9.0],
        [3.0, 8.0],
        [2.0, 11.0], # Dominated
        [0.5, 12.0]
    ])
    
    mask = compute_pareto_frontier(points)
    expected = np.array([True, True, True, False, True])
    
    np.testing.assert_array_equal(mask, expected)

def test_signed_dist():
    frontier = np.array([[1.0, 1.0]])
    
    # Worse
    p_worse = np.array([2.0, 2.0])
    d = signed_dist(p_worse, frontier)
    assert d > 0
    assert np.isclose(d, np.sqrt(2))
    
    # Better
    p_better = np.array([0.0, 0.0])
    d = signed_dist(p_better, frontier)
    assert d < 0
    assert np.isclose(d, -np.sqrt(2))
    
    # Mixed (Incomparable) -> Treated as positive in our logic
    p_mixed = np.array([0.5, 1.5])
    d = signed_dist(p_mixed, frontier)
    assert d > 0
