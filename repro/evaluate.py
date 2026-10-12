"""
Evaluate GraphPlanner checkpoints and simple baseline policies on a test file.

Policies:
    graphplanner  trained checkpoint (--model_path). Transductive by default:
                  uses the historical memory saved next to the checkpoint
                  (--memory_path to override); --inductive uses an empty memory.
    fixed         always answers the root query directly with one LLM (--llm);
                  the "single LLM, no routing" baseline.
    random        uniformly random valid (role, LLM) action at every step.

Results (per-query records + per-task/domain accuracy) are written to --out.

Examples:
    python -m repro.evaluate --policy fixed --llm gemma-3-12b-it --out results/fixed_gemma.json
    python -m repro.evaluate --policy random --out results/random.json
    python -m repro.evaluate --policy graphplanner \\
        --model_path checkpoints/main_best_model.pth --out results/graphplanner.json
"""

import argparse
import json
import os

import numpy as np
import torch

from router_planner.dynamic_router.train import (MemoryTable, load_best_model,
                                                 test_agent_on_dataset)


class RandomPolicy:
    """Uniformly random valid action; mimics RouteAgent.select_action's interface."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def select_action(self, state, local_tables, his_tables, his_index, action_mask=None, greedy=False):
        valid = np.flatnonzero(np.asarray(action_mask, dtype=float) > 0)
        return int(self.rng.choice(valid)), 0.0, 0.0, 0.0


class FixedLLMPolicy:
    """Always picks the executor role with one fixed LLM."""

    def __init__(self, llm_file, llm_name):
        llms = list(json.load(open(llm_file)).keys())
        if llm_name not in llms:
            raise ValueError(f"--llm must be one of {llms}")
        # Env_route builds actions as [planner x LLMs, executor x LLMs, summarizer x LLMs]
        self.action = len(llms) + llms.index(llm_name)

    def select_action(self, state, local_tables, his_tables, his_index, action_mask=None, greedy=False):
        assert action_mask[self.action] > 0, "executor action unexpectedly masked"
        return self.action, 0.0, 0.0, 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["graphplanner", "fixed", "random"], required=True)
    parser.add_argument("--llm", help="LLM key for --policy fixed")
    parser.add_argument("--model_path", help="checkpoint for --policy graphplanner")
    parser.add_argument("--memory_path", help="historical memory (default: derived from --model_path)")
    parser.add_argument("--inductive", action="store_true", help="evaluate with an empty historical memory")
    parser.add_argument("--gnn_backbone", default="dblayer", choices=["dblayer", "homo", "hetero"])
    parser.add_argument("--test_query_file", default="data/router_data_test_small.csv")
    parser.add_argument("--llm_file", default="config/llm_repro_phase2.json")
    parser.add_argument("--agent_file", default="config/agent_roles_with_embeddings.json")
    parser.add_argument("--max_planner_calls", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    memory = MemoryTable()
    if args.policy == "graphplanner":
        agent = load_best_model(args.model_path, embed_dim=768, candidate_embed_dim=768 * 2,
                                hidden_dim=32, gnn_backbone=args.gnn_backbone)
        if not args.inductive:
            memory_path = args.memory_path or args.model_path.replace("_model.pth", "_memory.pkl")
            memory.load_from_file(memory_path)
    elif args.policy == "fixed":
        agent = FixedLLMPolicy(args.llm_file, args.llm)
    else:
        agent = RandomPolicy(args.seed)

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    stats = test_agent_on_dataset(
        agent=agent,
        test_query_file=args.test_query_file,
        llm_file=args.llm_file,
        agent_file=args.agent_file,
        max_planner_calls=args.max_planner_calls,
        batch_size=args.batch_size,
        his_memory=memory,
        save_path=args.out,
    )

    print(f"\nAcc = {stats['test_acc']:.3f}  Avg cost = {stats['test_avg_cost']:.1f}  "
          f"Avg LLM calls = {stats['test_avg_llm_calls']:.2f}")
    for k, v in stats.items():
        if k.startswith("test_acc_domain/"):
            print(f"  {k.split('/')[1]:8s} {v:.3f}")


if __name__ == "__main__":
    main()
