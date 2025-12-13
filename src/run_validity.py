import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from utils import *
import random
import time



if __name__ == "__main__":
    if len(sys.argv) > 1:
        num_trials = int(sys.argv[1])
    else:
        num_trials = 10  # default

    n = 30
    p = 10
    sigma = 1.0
    K = 3
    tau_list = [ 0.1, 0.25, 0.5, 1, 5]
    linkage = "complete"
    #num_trials = 1000
    n_jobs = -1

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(base_dir, "results/raw")
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.perf_counter()
    all_p_values, naive_p_values = check_p_value_uniformity_multi_tau_parallel(
        n, p, sigma, K, tau_list, linkage, num_trials, n_jobs
    )
    end_time = time.perf_counter()  # ---- end timing ----
    print(f"Total runtime: {end_time - start_time:.2f} seconds")

    df = pd.DataFrame({f"tau={tau}": all_p_values[tau] for tau in tau_list})
    df["naive"] = naive_p_values
    csv_path = os.path.join(output_dir, f"pval_validity_randomized_K{K}_ntrials{num_trials}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")


