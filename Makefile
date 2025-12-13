# ============================================================
# Makefile for FDR Simulation Project (BH 1995 reproduction)
# ============================================================

# Variables
PYTHON := python
SRC_DIR := src
RESULTS_DIR := results
RAW_DIR := $(RESULTS_DIR)/raw
FIG_DIR := $(RESULTS_DIR)/figs
SIM_SCRIPT := $(SRC_DIR)/run_validity.py
TEST_DIR := tests
PROFILE_SCRIPT = profiling/profile_test.py
PROFILE_OUT = $(RESULTS_DIR)/raw/prof_merge.prof

# ============================================================
# Targets
# ============================================================
# 1. Run the full pipeline
all: simulate figures
	@echo "All tasks completed successfully."

# 2. Run simulations and save raw results
simulate:
	@echo "Running simulations..."
	$(PYTHON)  -m src.run_validity
	@echo "Simulation complete. Results saved in $(RAW_DIR)"

# 3. Create visualizations
figures:
	@echo "Generating figures..."
	$(PYTHON) -m src.plot_ecdf
	@echo "Figures saved in $(FIG_DIR)"

# 4. Remove generated files
clean:
	@echo "Cleaning up results..."
	rm -rf $(RESULTS_DIR)
	@echo "All generated files removed."

# 5. Run test suite (if you have tests/)
test:
	@echo "Running tests..."
	pytest $(TEST_DIR) -v
	@echo "All tests passed."

# 6. Profiling
profile:
	$(PYTHON) -m cProfile -o $(PROFILE_OUT) $(PROFILE_SCRIPT)
	@echo "Profile saved to $(PROFILE_OUT)"
	@echo "Launching SnakeViz..."
	snakeviz $(PROFILE_OUT)
 