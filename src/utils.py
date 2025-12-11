import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hierarchical_clustering_invariant import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal,f
from joblib import Parallel, delayed
from itertools import combinations
from dgps import *

def compute_nu(node,n):
    # return the projection direction from the given node
    G_1= np.array(node.left.points)
    G_2 = np.array(node.right.points)
    n_G1 = len(G_1)
    n_G2 = len(G_2)

    nu = np.zeros(n)

    nu[G_1] += 1/n_G1
    nu[G_2] -= 1/n_G2
    return nu

def compute_nu_pair(c1,c2, n):
    # return the projection direction from the given node
    G_1 = np.array(c1.points)
    G_2 = np.array(c2.points)
    n_G1 = len(G_1)
    n_G2 = len(G_2)

    nu = np.zeros(n)

    nu[G_1] += 1 / n_G1
    nu[G_2] -= 1 / n_G2
    return nu

def create_indicator_diagonal_matrix(index_list, n):
    diag = np.zeros(n)
    diag[index_list] = 1
    return np.diag(diag), diag


def naive_p_value(X, K, layer, linkage):
    n = X.shape[0]
    p = X.shape[1]
    model = AgglomerativeClustering(X, tau=0, n_clusters=K, linkage=linkage) #fit non-randomized hierarchical clustering model
    model.fit()
    winning_nodes = list(model.existing_clusters_log.keys())
    key = winning_nodes[layer]
    node = key[0].parent
    nu = compute_nu(node, n).reshape(-1, 1)
    p_node_1 = node.left
    p_node_2 = node.right
    m = len(p_node_1.points) + len(p_node_2.points)
    c1 = model.K_clusters[0]
    c2 = model.K_clusters[1]
    P0 = nu @ nu.T / np.linalg.norm(nu) ** 2
    I1, one1 = create_indicator_diagonal_matrix(p_node_1.points, n)
    I2, one2 = create_indicator_diagonal_matrix(p_node_2.points, n)
    one1 = one1.reshape(-1, 1)
    one2 = one2.reshape(-1, 1)
    P1 = (I1 - one1 @ one1.T / len(p_node_1.points)) + (I2 - one2 @ one2.T / len(p_node_2.points))

    observed_target = (m - 2) * np.linalg.norm(P0 @ X, 'fro') ** 2 / np.linalg.norm(P1 @ X, 'fro') ** 2

    p_value = 1 - f.cdf(observed_target, dfn=p, dfd=(m - 2) * p)
    return p_value

def naive_p_value_random_pair(X, K, linkage):
    n = X.shape[0]
    p = X.shape[1]
    model = AgglomerativeClustering(X, tau=0, n_clusters=K, linkage=linkage) #fit non-randomized hierarchical clustering model
    model.fit()
    c1 = model.K_clusters[0]
    c2 = model.K_clusters[1]
    nu = compute_nu_pair(c1,c2,n).reshape(-1, 1)
    p_node_1 = c1
    p_node_2 = c2
    m = len(p_node_1.points) + len(p_node_2.points)
    if m <= 2:
        return np.nan
    P0 = nu @ nu.T / np.linalg.norm(nu) ** 2
    I1, one1 = create_indicator_diagonal_matrix(p_node_1.points, n)
    I2, one2 = create_indicator_diagonal_matrix(p_node_2.points, n)
    one1 = one1.reshape(-1, 1)
    one2 = one2.reshape(-1, 1)
    P1 = (I1 - one1 @ one1.T / len(p_node_1.points)) + (I2 - one2 @ one2.T / len(p_node_2.points))

    observed_target = (m - 2) * np.linalg.norm(P0 @ X, 'fro') ** 2 / np.linalg.norm(P1 @ X, 'fro') ** 2

    p_value = 1 - f.cdf(observed_target, dfn=p, dfd=(m - 2) * p)
    return p_value


def check_p_value_uniformity(n, p, sigma, K, tau, layer, linkage="complete", num_trials=1000):
    p_values = []
    p_values_n = []
    mu = np.zeros(p)

    while len(p_values_n) < num_trials:
        X = generate_null_data(n, p, mu, sigma)
        model = AgglomerativeClustering(X, tau=tau, n_clusters=K, linkage=linkage)
        model.fit()

        winning_nodes = list(model.existing_clusters_log.keys())
        key = winning_nodes[layer]
        node = key[0].parent
        #c1 = model.K_clusters[0]
        #c2 = model.K_clusters[1]
        #p_value, obs, sel_corrected = model.merge_inference_F_random_pair_grid(c1,c2, grid_width=70, ncoarse=20, ngrid=2000)
        p_value, obs, sel_corrected = model.merge_inference_F_grid(node, grid_width=180, ncoarse=20,ngrid=2000)
        p_value_n = naive_p_value(X, K, layer, linkage)
        if not (np.isnan(p_value) and np.isnan(p_value_n)):
            p_values.append(p_value)
            p_values_n.append(p_value_n)
    p_values = np.array(p_values)
    p_values_n = np.array(p_values_n)

    # Histogram for both p-values
    plt.figure(figsize=(8, 5))
    plt.hist(p_values, bins=20, density=True, alpha=0.5, color="blue", edgecolor="black",
             label="Selection-based p-value")
    plt.hist(p_values_n, bins=20, density=True, alpha=0.5, color="orange", edgecolor="black", label="Naive p-value")
    plt.axhline(1, color='red', linestyle='dashed', linewidth=2, label="Uniform(0,1)")
    plt.xlabel("P-value")
    plt.ylabel("Density")
    plt.title("Histogram of P-values Under the Null Hypothesis")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.ecdfplot(p_values, color="blue", label="Selection-based p-values")
    sns.ecdfplot(p_values_n, color="orange", label="Naive p-values")
    plt.plot([0, 1], [0, 1], linestyle="--", color="red", label="Expected (Uniform)")
    plt.xlabel("P-value")
    plt.ylabel("ECDF")
    plt.title("Empirical CDF of P-values")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    # Q-Q Plot for both p-values
    plt.figure(figsize=(8, 5))
    sorted_p_values = np.sort(p_values)
    sorted_p_values_n = np.sort(p_values_n)
    theoretical_quantiles = np.linspace(0, 1, num_trials)

    plt.plot(theoretical_quantiles, sorted_p_values, marker='o', linestyle='', color="blue",
             label="Selection-based p-values")
    plt.plot(theoretical_quantiles, sorted_p_values_n, marker='o', linestyle='', color="orange", label="Naive p-values")
    plt.plot([0, 1], [0, 1], linestyle="--", color="red", label="Expected (Uniform)")
    plt.xlabel("Theoretical Uniform Quantiles")
    plt.ylabel("Empirical P-values")
    plt.title("Q-Q Plot: P-values vs. Uniform(0,1)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

