import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results")))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_ecdf_combined(df, tau_cols, naive, output_dir, K):
    """Plot combined ECDF figure for all tau values + naive."""
    def plot_ecdf(values, label, color=None):
        x = np.sort(values)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where='post', label=label, color=color)

    # color map
    cmap = plt.cm.GnBu
    rand_colors = [cmap(i) for i in np.linspace(0.2, 0.9, len(tau_cols))]
    color_map = {
        "Naive": "#FF758F",
        "Expected (Uniform)": "#FF0000",
    }

    plt.figure(figsize=(10, 6))
    for i, col in enumerate(tau_cols):
        tau_val = col.split('=')[1]
        label = f"RAC({tau_val})"
        color = rand_colors[i]
        plot_ecdf(df[col].dropna().to_numpy(), label=label, color=color)

    # naive
    plot_ecdf(naive, "Naive", color=color_map["Naive"])

    # expected uniform line
    plt.plot([0, 1], [0, 1], linestyle="--", label="Expected (Uniform)", color=color_map["Expected (Uniform)"])

    plt.xlabel("P-value")
    plt.ylabel("ECDF")
    plt.title(f"ECDF of P-values (K={K})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"ecdf_combined_K{K}.png")
    plt.savefig(save_path)
    print(f"Saved ECDF plot to {save_path}")



if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(root, "results", "raw")
    output_dir = os.path.join(root, "results", "figs")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(raw_dir, "pval_validity_randomized_K3.csv"))
    tau_cols = [c for c in df.columns if c.startswith('tau=')]

    naive = df['naive'].dropna().to_numpy()
    K = 3

    # Call plotting functions
    plot_ecdf_combined(df, tau_cols, naive, output_dir, K)
