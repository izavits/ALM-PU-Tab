# ALM-PU-Tab

ALM-PU applied to tabular PU benchmarks with a small MLP. It is evaluated in several splits across labeling mechanisms, label proportions and and metrics.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Versions are pinned to those used for the reported results (Python 3.14). The code runs on CPU and uses a GPU automatically if one is available.

## Running the benchmark

```bash
.venv/bin/python pu_benchmark.py
```

This runs every dataset in `dataset/` under SCAR and SAR labeling, hiding 25%, 50% and 75% of the training positives, with 10 runs per setting (run *i* uses split `random_state=i`). Settings that already have results are skipped; use `--overwrite` to rerun them.


## Datasets

The 14 datasets in `dataset/` are headerless, standardized CSV files with a binary label in the last column; 1 is the positive (minority) class.

## Reference
This is based on the following work and cloned/modified from the respective code repo:

Wei, J., Wu, Y., Shi, B. et al. ALM-PU: positive and unlabeled learning with constrained optimization. Mach Learn 114, 210 (2025). https://doi.org/10.1007/s10994-025-06849-3
