#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
config.py — central global settings for the RoChBert reproduction pipeline.

Constants that define a run, kept in one place so seeds and hyperparameters do
not drift across scripts.
CLI flags still override these per run; these are the defaults.
"""

# Reproducibility -----------------------------------------------------------
# Global run seed. Trainers historically used 2333 (utils/random_seed.py); we
# keep it so training reproduces. Per-example seeds are derived from this via
# repro_utils.seed_for_example so runs are restart-safe.
SEED = 2333

# Attack / augmentation -----------------------------------------------------
# Substitute candidate cap. Paper Appendix A.2: "at most 40 candidate words".
ATTACK_THRESHOLD = 40

# Number of test examples attacked. Paper §4.1: "1,000 texts sampled from test set".
DATA_NUM = 1000

# Max modification rate for data augmentation, per dataset. Paper Table 6.
# eps_max is applied to the FINAL adversarial text of a trace (Algorithm 1).
EPS_MAX = {
    "ChnSentiCorp": 0.45,
    "DMSC": 0.45,
    "THUCNews": 0.30,
    "OCNLI": 0.30,          # Random attack on OCNLI uses 0.10 (Table 6)
}
DEFAULT_EPS_MAX = 0.45      # ChnSentiCorp default

# Max input length (chars) an example is eligible for (matches attack scripts).
MAX_LEN = 512
