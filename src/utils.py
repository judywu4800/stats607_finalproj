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


def naive_p_value(X, K, linkage="complete"):
    n = X.shape[0]
    p = X.shape[1]
    model = AgglomerativeClustering(X, tau=0, n_clusters=K, linkage=linkage) #fit non-randomized hierarchical clustering model
    model.fit()
    winning_nodes = list(model.existing_clusters_log.keys())
    key = winning_nodes[-1]
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

def check_p_value_uniformity(n, p, sigma, K, tau, linkage="complete", num_trials=1000):
    p_values = []
    p_values_n = []
    mu = np.zeros(p)

    while len(p_values_n) < num_trials:
        X = generate_null_data(n, p, mu, sigma)
        model = AgglomerativeClustering(X, tau=tau, n_clusters=K, linkage=linkage)
        model.fit()

        winning_nodes = list(model.existing_clusters_log.keys())
        key = winning_nodes[-1]
        node = key[0].parent
        p_value, obs, sel_corrected = model.merge_inference_F_grid(node, grid_width=180, ncoarse=20,ngrid=2000)
        p_value_n = naive_p_value(X, K, linkage)
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



def run_trial(n, p, sigma, K, tau_list, linkage):
    X = generate_null_data(n, p, np.zeros(p), sigma)
    trial_results = {}

    naive_val = naive_p_value(X, K, linkage)
    trial_results['naive'] = naive_val

    for tau in tau_list:
        model = AgglomerativeClustering(X, tau=tau, n_clusters=K, linkage=linkage)
        model.fit()
        winning_nodes = list(model.existing_clusters_log.keys())
        key = winning_nodes[-1]
        node = key[0].parent
        p_val, _, _ = model.merge_inference_F_grid(node, grid_width=100, ncoarse=20, ngrid=2000)
        trial_results[tau] = p_val

    return trial_results

def check_p_value_uniformity_multi_tau_parallel(n, p, sigma, K, tau_list,
                                                linkage="complete", num_trials=1000, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_trial)(n, p, sigma, K, tau_list, linkage)
        for _ in range(num_trials)
    )

    all_p_values = {tau: [] for tau in tau_list}
    naive_p_values = []

    for res in results:
        naive_p_values.append(res['naive'])
        for tau in tau_list:
            all_p_values[tau].append(res[tau])

    for tau in tau_list:
        all_p_values[tau] = np.array(all_p_values[tau])
    naive_p_values = np.array(naive_p_values)

    return all_p_values, naive_p_values

def single_repeat(tau, label, n, p, sigma, K, layer, alpha, num_trials):
    mu = np.zeros(p)
    p_values = []

    while len(p_values)<num_trials:
        X = generate_null_data(n, p, mu, sigma)
        model = AgglomerativeClustering(X, tau=tau, n_clusters=K, linkage="complete")
        model.fit()

        winning_nodes = list(model.existing_clusters_log.keys())
        key = winning_nodes[layer]
        node = key[0].parent

        p_val, _, _ = model.merge_inference_F(node, grid_width=5, ncoarse=20, ngrid=1000)
        if not np.isnan(p_val):
            p_values.append(p_val)

    type_I_error = np.mean(np.array(p_values) < alpha)
    return {"Tau": tau, "Type": label, "Type I Error": type_I_error}


def check_type1_multi_tau_parallel(n, p, sigma, tau_list, K, layer, alpha=0.05, num_trials=200, num_repeats=10,
                                   n_jobs=-1):
    tasks = []
    for tau in tau_list:
        label = "Naive" if tau == 0 else "Randomized"
        for _ in range(num_repeats):
            tasks.append((tau, label, n, p, sigma, K, layer, alpha, num_trials))

    results = Parallel(n_jobs=n_jobs)(
        delayed(single_repeat)(*task) for task in tasks
    )

    df_results = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_results, x="Tau", y="Type I Error", hue="Type")
    plt.axhline(y=alpha, linestyle='--', color='red', label   =f"Significance level α = {alpha}")
    plt.title(f"Distribution of Type I Error Rates over {num_repeats} Repetitions")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()
    return df_results