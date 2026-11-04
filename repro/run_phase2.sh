#!/usr/bin/env bash
# Phase 2 main experiments (DeepInfra pool, see repro/build_config.py).
#
# Stage "train": GraphPlanner, 3 seeds, plus the w/o History ablation (seed 0),
#   1000 episodes x batch 1 each, run in parallel. Meanwhile the single-LLM
#   and random baselines are evaluated on the full 556-query test set, followed
#   by stage "labels" (sequential, to stay within 15 GB of RAM).
# Stage "labels": single-call labels of every LLM on 1000 training queries,
#   for the Router-KNN / GraphRouter baselines.
# Stage "eval": transductive and inductive evaluation of every checkpoint.
#
# Finished trainings and evaluations whose output file already exists are skipped,
# and unfinished trainings resume from their last checkpoint, so after an
# interruption the same command continues where it stopped. All processes share cache/api_spend.sqlite (budget LLM_BUDGET_USD, default $8)
# and cache/llm_responses.sqlite. Usage: bash repro/run_phase2.sh {train|labels|eval}
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphplanner
eval "$(grep '^export DEEPINFRA_API_KEY=' ~/.bashrc)"
export HF_HUB_OFFLINE=1 WANDB_MODE=offline OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
mkdir -p logs results checkpoints_phase2

TEST=data/router_data_test.csv
LLMS="llama-3.1-8b-instruct mistral-nemo-12b-instruct llama-3.3-70b-instruct mistral-small-24b-instruct"

train() {  # name, extra args...
    local name=$1; shift
    [ -f "checkpoints_phase2/${name}_final_model.pth" ] && return 0  # already trained
    # --resume continues from checkpoints_phase2/${name}_resume.pt (written every 25 episodes) if present
    python -m router_planner.dynamic_router.train \
        --query_file data/router_data_train_clean.csv --test_query_file data/router_data_test_small.csv \
        --llm_file config/llm_repro_phase2.json --agent_file config/agent_roles_with_embeddings.json \
        --max_planner_calls 2 --max_episodes 1000 --batch_size 1 --test_every_n_episodes 100000 \
        --save_dir checkpoints_phase2 --experiment_name "$name" --resume "$@" >> "logs/train_$name.log" 2>&1
}

labels() {
    # Single-call labels for the Router-KNN / GraphRouter baselines: every LLM answers
    # 1000 training queries drawn uniformly (as route_env samples them). Test-set labels
    # are the fixed-policy results of stage "train".
    python - <<'EOF'
import pandas as pd
pd.read_csv("data/router_data_train_clean.csv").sample(1000, random_state=0) \
    .to_csv("data/router_labels_train1000.csv", index=False)
EOF
    for llm in $LLMS; do
        [ -f "results/labels_train1000_$llm.json" ] && continue  # already labelled
        python -m repro.evaluate --policy fixed --llm "$llm" --test_query_file data/router_labels_train1000.csv \
            --batch_size 8 --out "results/labels_train1000_$llm.json" > "logs/labels_train1000_$llm.log" 2>&1
    done
}

case "${1:-}" in
train)
    for s in 0 1 2; do train "gp_seed$s" --seed "$s" & done
    train gp_nohist_seed0 --seed 0 --no_history &
    (
        for llm in $LLMS; do
            [ -f "results/fixed_$llm.json" ] && continue  # already evaluated
            python -m repro.evaluate --policy fixed --llm "$llm" --test_query_file $TEST --batch_size 8 \
                --out "results/fixed_$llm.json" > "logs/eval_fixed_$llm.log" 2>&1
        done
        [ -f results/random.json ] || python -m repro.evaluate --policy random --seed 0 --test_query_file $TEST \
            --batch_size 4 --out results/random.json > logs/eval_random.log 2>&1
        labels
    ) &
    wait
    ;;
labels)
    labels
    ;;
eval)
    for name in gp_seed0 gp_seed1 gp_seed2 gp_nohist_seed0; do
        (
            ckpt=checkpoints_phase2/${name}_best_model.pth
            python -m repro.evaluate --policy graphplanner --model_path "$ckpt" --test_query_file $TEST \
                --batch_size 4 --out "results/${name}_trans.json" > "logs/eval_${name}_trans.log" 2>&1
            python -m repro.evaluate --policy graphplanner --model_path "$ckpt" --inductive --test_query_file $TEST \
                --batch_size 4 --out "results/${name}_ind.json" > "logs/eval_${name}_ind.log" 2>&1
        ) &
    done
    wait
    ;;
*)
    echo "usage: $0 {train|labels|eval}" >&2; exit 1 ;;
esac
