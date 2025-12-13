import numpy as np
import sys, os
import pytest
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from hierarchical_clustering_invariant import AgglomerativeClustering


@pytest.fixture
def example_model():
    """
    Build a small example RAC model.
    This fixture is reused across multiple tests.
    """
    np.random.seed(0)
    X = np.random.randn(30, 5)

    model = AgglomerativeClustering(X, n_clusters=2, tau=0.1)
    model.fit()
    return model


@pytest.fixture
def last_merge_node(example_model):
    """Return the node used for inference (final merge)."""
    model = example_model
    winning_nodes = list(model.existing_clusters_log.keys())
    key = winning_nodes[-1]
    return key[0].parent


def compute_pvalue(model, node, ncoarse=20):
    """Helper to compute p-value once."""
    pval,_,_= model.merge_inference_F(node, ngrid=2000, ncoarse=ncoarse, grid_width=10)
    return pval

def compute_pvalue_grid(model, node, ncoarse=20):
    """Helper to compute p-value once."""
    pval,_,_= model.merge_inference_F_grid(node, ngrid=2000, ncoarse=ncoarse, grid_width=10)
    return pval


def test_pvalue_runs(example_model, last_merge_node):
    """Test that the p-value computation completes without errors."""
    p = compute_pvalue(example_model, last_merge_node)
    assert p is not None, "merge_inference_F returned None"


def test_pvalue_type(example_model, last_merge_node):
    """Test that the p-value is a float."""
    p = compute_pvalue(example_model, last_merge_node)
    assert isinstance(p, float) or isinstance(p, np.floating), \
        f"Expected float p-value, got {type(p)}"


def test_pvalue_range(example_model, last_merge_node):
    """Test that p-value lies within [0, 1]."""
    p = compute_pvalue(example_model, last_merge_node)
    assert 0.0 <= p <= 1.0, f"p-value out of range: {p}"

def test_pvalue_type_grid(example_model, last_merge_node):
    """Test that the p-value is a float."""
    p = compute_pvalue_grid(example_model, last_merge_node)
    assert isinstance(p, float) or isinstance(p, np.floating), \
        f"Expected float p-value, got {type(p)}"


def test_pvalue_range_grid(example_model, last_merge_node):
    """Test that p-value lies within [0, 1]."""
    p = compute_pvalue_grid(example_model, last_merge_node)
    assert 0.0 <= p <= 1.0, f"p-value out of range: {p}"


def test_reproducibility():
    """
    Test reproducibility under fixed RNG.
    This detects hidden randomness in the pipeline.
    """
    np.random.seed(0)
    X = np.random.randn(30, 5)
    model1 = AgglomerativeClustering(X.copy(), n_clusters=2, tau=0.1, random_state=0)
    model1.fit()
    node1 = list(model1.existing_clusters_log.keys())[-1][0].parent
    p1 = compute_pvalue(model1, node1)

    np.random.seed(0)
    X = np.random.randn(30, 5)
    model2 = AgglomerativeClustering(X.copy(), n_clusters=2, tau=0.1,random_state=0)
    model2.fit()
    node2 = list(model2.existing_clusters_log.keys())[-1][0].parent
    p2 = compute_pvalue(model2, node2)

    assert abs(p1 - p2) < 1e-12, f"P-values not reproducible: {p1} vs {p2}"

def test_coarse_grid_accuracy(example_model, last_merge_node):
    """
    Test that the p-value computed using a coarse grid is close
    to the p-value computed using a fine grid.
    """

    model = example_model
    node = last_merge_node

    # fine grid (baseline)
    np.random.seed(0)
    p_fine = compute_pvalue(model, node, ncoarse=None)

    # coarse grid (approximation)
    np.random.seed(0)
    p_coarse = compute_pvalue(model, node, ncoarse=20)

    # they should be numerically close
    assert abs(p_fine - p_coarse) < 0.05, \
        f"Coarse grid p-value {p_coarse} deviates too much from fine grid {p_fine}"
