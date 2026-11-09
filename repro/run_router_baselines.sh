#!/usr/bin/env bash
# Offline single-round router baselines from the single-call labels written by
# repro/run_phase2.sh (stages "train" and "labels"): Router-KNN, best single
# LLM and oracle (repro/router_baselines.py), GraphRouter with 3 seeds
# (repro/graphrouter_baseline.py) and RouterDC with 3 seeds
# (repro/routerdc_baseline.py, about 3 h per seed on CPU, resumable). No API calls.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphplanner
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4

python -m repro.router_baselines --out results/router_baselines.json 2>&1 | tee logs/router_baselines.log
python -m repro.graphrouter_baseline --seeds 0 1 2 --out results/graphrouter.json 2>&1 \
    | grep -v "result_predict" | tee logs/graphrouter.log
OMP_NUM_THREADS=10 python -m repro.routerdc_baseline --seeds 0 1 2 --out results/routerdc.json 2>&1 \
    | grep -v -i "warn" | tee -a logs/routerdc.log
