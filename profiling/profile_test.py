import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from datetime import datetime
import random
from hierarchical_clustering_invariant import AgglomerativeClustering
from dgps import generate_null_data

np.random.seed(0)
X = generate_null_data(30,2,np.zeros(2))

def run_full_inference():
    model = AgglomerativeClustering(X, n_clusters=2, tau=0.1)
    model.fit()

    winning_nodes = list(model.existing_clusters_log.keys())
    key = winning_nodes[-1]
    node = key[0].parent
    model.merge_inference_F(node, ngrid=2000,ncoarse=None, grid_width=15)

if __name__ == "__main__":
    run_full_inference()
