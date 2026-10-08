import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

from src.config import Config
from src.experiment import run_experiments


# Main File for Experiments
#
# Edit CONFIGS below: one entry per evaluation run. Every entry starts from
# BASE (defaults live in src/config.py) and only overrides what differs.
# Runs execute one after another; each writes results/<name>.json,
# results/<name>_samples.json, and results/<name>-server.log.

BASE = Config()

CONFIGS = [
    BASE.with_overrides(name="math-q4",num_examples=50),
    BASE.with_overrides(name="code-q4",num_examples=50),
]


if __name__ == "__main__":
    run_experiments(CONFIGS)