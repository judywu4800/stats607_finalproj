# stats607_finalproj

The project repository contains all source code, documentation, and example workflows needed to reproduce the results and run the demo. 

This folder is organized as follows.
```
stats607_finalproj
├── doc
│   ├── demo_notebook.ipynb
│   └── report_Wu.md
├── Makefile
├── profiling
│   └── profile_test.py
├── README.md
├── requirements.txt
├── results
│   ├── figs
│   │   ├── ecdf_combined_K3.png
│   │   ├── profiling.png
│   │   └── qq_plot_combined_K3.png
│   └── raw
│       ├── prof_merge.prof
│       ├── pval_validity_randomized_K3_ntrials10.csv
│       └── pval_validity_randomized_K3.csv
├── src
│   ├── __pycache__
│   │   ├── dgps.cpython-310.pyc
│   │   ├── hierarchical_clustering_invariant.cpython-310.pyc
│   │   ├── plot_ecdf.cpython-310.pyc
│   │   ├── run_validity.cpython-310.pyc
│   │   └── utils.cpython-310.pyc
│   ├── dgps.py
│   ├── hierarchical_clustering_invariant.py
│   ├── plot_ecdf.py
│   ├── run_validity.py
│   └── utils.py
└── tests
    ├── __pycache__
    │   └── test_pval.cpython-310-pytest-9.0.2.pyc
    └── test_pval.py

```

* Source code and simulation codes are saved under `src/`.

* Intermediate and final outputs are saved under:
   ```bash
  results/raw
  results/figures
   ```

* Reports and demo notebook are saved under `doc/`.



## Setup Instructions

1. **Clone this repository**
   ```bash
   git clone git@github.com:judywu4800/stats607_finalproj.git
   cd stats607_finalproj
   ```

2. **Create and activate a virtual environment**
    ```bash
    conda create -n rac-env python=3.10
    conda activate rac-env
    ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

# Run the Complete Analysis
- `make all`: Run simulations, save raw results and produce figures.
- `make simulate`: Run simulations and save raw results.
- `make figures`: Generate plots.
- `make profile`: Run profiling on inference pipeline and produce `snakeviz` visualization.
- `make test`: Run tests.
- `make clean`: Clean all results.


