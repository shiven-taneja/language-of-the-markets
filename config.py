from pathlib import Path

# Random seeds for reproducibility
SEEDS = [8, 26, 1111]

# Tickers to process
# TICKERS = ["DIS", "BABA", "GOOG", "KO", "MRK", "MS", "NVDA", "QQQ", "T", "WFC"]
TICKERS = ["GE"]

# Experiment types
RUN_TYPES = ["baseline", "headline", "techsent", "all"]

# Paths
DATA_ROOT = Path("data")
RESULTS_ROOT = Path("results")
CHECKPOINTS_ROOT = Path("checkpoints")

# Evaluation settings
SELECTION_METRIC = "final_return_%"
