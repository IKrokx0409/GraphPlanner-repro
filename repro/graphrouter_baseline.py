"""
GraphRouter baseline (Feng et al., 2024) on the single-call labels.

The GNN, its training loop and its hyperparameters are the official ones
(repro/third_party/graphrouter/graph_nn.py, vendored unchanged; defaults from
the official configs/config.yaml). Adaptations to our setting:

- Graph: query nodes are the 1000 labelled training queries plus the 556 test
  queries, LLM nodes are our 4 LLMs (description embeddings from
  config/llm_repro_phase2.json), and every query-LLM edge carries the utility
  ("effect") and the cost, normalised per task by min-max over the training
  edges (the official data construction also normalises cost per task).
  Task nodes use short task descriptions embedded with the same Longformer.
- Split: our fixed train/test files instead of the official per-task ratio
  split; 10% of the training queries per task are held out for validation.
- The test-time routing decision comes from the checkpoint with the best
  validation macro-F1 (the official loop saves it but reports the last epoch).
- LLMs are ordered by nominal price, so the argmax label of a query on which
  several LLMs are correct is the cheapest of them (same tie rule as Router-KNN).

Scenario "Performance First" (effect = utility), as for the other
single-round routers in Table 3.

Usage:
    python -m repro.graphrouter_baseline [--seeds 0 1 2] [--out results/graphrouter.json]
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from repro.router_baselines import LLMS, load_labels, summarize
from repro.third_party.graphrouter.graph_nn import GNN_prediction, form_data

OFFICIAL_CONFIG = {  # GraphRouter configs/config.yaml
    "train_epoch": 1000, "learning_rate": 0.0003, "weight_decay": 0.0001, "train_mask_rate": 0.5,
    "batch_size": 32, "embedding_dim": 8, "edge_dim": 3,
}

# Written in the style of the official task descriptions (data_processing/multidata_unify.py)
TASK_DESCRIPTIONS = {
    "gsm8k": "The GSM8K dataset is tailored for mathematical problem-solving tasks. It consists of grade-school "
             "math word problems that require multi-step arithmetic reasoning to reach a numeric answer.",
    "math": "The MATH dataset contains competition mathematics problems in algebra, geometry, number theory and "
            "other areas, requiring advanced multi-step reasoning and an exact final answer.",
    "mbpp": "The MBPP dataset is designed for Python code generation. The model writes a short function from a "
            "natural language description, and the code is checked against test assertions.",
    "human_eval": "The HumanEval dataset evaluates functional correctness of code generation. The model completes a "
                  "Python function from its signature and docstring, and the completion is run against unit tests.",
    "commonsense_qa": "The CommonsenseQA dataset is a multiple-choice question answering task that requires "
                      "everyday commonsense knowledge about concepts and their relations.",
    "arc_challenge": "The ARC-Challenge dataset contains difficult grade-school science multiple-choice questions "
                     "that require scientific knowledge and reasoning.",
    "openbook_qa": "The OpenBookQA dataset contains elementary science multiple-choice questions that require "
                   "combining a core science fact with broad common knowledge.",
    "natural_qa": "The Natural Questions dataset is an open-domain question answering task built from real search "
                  "queries, where the model must produce a short factual answer from world knowledge.",
    "trivia_qa": "The TriviaQA dataset is an open-domain question answering task with trivia questions, where the "
                 "model must recall precise facts about entities and events.",
    "mmlu": "The MMLU dataset is a multiple-choice benchmark covering 57 academic and professional subjects, "
            "testing broad knowledge and problem-solving ability.",
    "gpqa": "The GPQA dataset contains graduate-level multiple-choice questions in biology, physics and chemistry "
            "written by domain experts, requiring deep expert reasoning.",
}


class _NoWandb:
    def log(self, *args, **kwargs):
        pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_graph(tr_data, tr_emb, tr_u, tr_c, te_data, te_emb, te_u, te_c, task_emb, llm_emb, val_frac, seed):
    tasks = np.concatenate([tr_data["task_name"].values, te_data["task_name"].values])
    emb = np.concatenate([tr_emb, te_emb])
    utility = np.concatenate([tr_u, te_u])
    cost = np.concatenate([tr_c, te_c])
    n_tr, n_q, m = len(tr_data), len(tasks), utility.shape[1]

    # Per-task min-max cost normalisation, fitted on training edges only
    norm_cost = np.zeros_like(cost)
    for t in np.unique(tasks):
        rows = tasks == t
        ref = cost[:n_tr][tasks[:n_tr] == t]
        lo, hi = ref.min(), ref.max()
        norm_cost[rows] = np.clip((cost[rows] - lo) / (hi - lo if hi > lo else 1.0), 0, 1)

    rng = np.random.RandomState(seed)
    val_q = []
    for t in np.unique(tasks[:n_tr]):
        q = np.flatnonzero(tasks[:n_tr] == t)
        val_q.extend(rng.choice(q, max(1, int(round(len(q) * val_frac))), replace=False))
    split = np.zeros(n_q, dtype=int)  # 0 train, 1 validation, 2 test
    split[val_q] = 1
    split[n_tr:] = 2
    edge_split = np.repeat(split, m)

    effect = utility.reshape(-1)  # Performance First
    label = np.eye(m)[np.argmax(utility, axis=1)].reshape(-1, 1)
    org = [q for q in range(n_q) for _ in range(m)]
    des = list(range(m)) * n_q
    combined = np.stack([norm_cost.reshape(-1), effect], axis=1)
    masks = [torch.tensor(edge_split == s, dtype=torch.float) for s in range(3)]
    task_feat = np.stack([task_emb[t] for t in tasks])

    fd = form_data("cpu")
    graphs = [fd.formulation(task_id=task_feat, query_feature=emb, llm_feature=llm_emb, org_node=org,
                             des_node=des, edge_feature=effect, label=label, edge_mask=mk,
                             combined_edge=combined, train_mask=masks[0], valide_mask=masks[1],
                             test_mask=masks[2]) for mk in masks]
    return graphs, masks


def route(gnn, graph, masks, m):
    """LLM index per test query from the current model (sees train and validation edges)."""
    gnn.model.eval()
    with torch.no_grad():
        pred = gnn.model(task_id=graph.task_id, query_features=graph.query_features,
                         llm_features=graph.llm_features, edge_index=graph.edge_index,
                         edge_mask=masks[2].bool(), edge_can_see=(masks[0] + masks[1]).bool(),
                         edge_weight=graph.combined_edge)
    return torch.argmax(pred.reshape(-1, m), dim=1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/router_labels_train1000.csv")
    parser.add_argument("--train_results", default="results/labels_train1000_{llm}.json")
    parser.add_argument("--test_csv", default="data/router_data_test.csv")
    parser.add_argument("--test_results", default="results/fixed_{llm}.json")
    parser.add_argument("--llm_file", default="config/llm_repro_phase2.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--ckpt_dir", default="checkpoints_phase2/graphrouter")
    parser.add_argument("--out", default="results/graphrouter.json")
    args = parser.parse_args()

    tr_data, tr_emb, tr_u, tr_c = load_labels(args.train_csv, args.train_results)
    te_data, te_emb, te_u, te_c = load_labels(args.test_csv, args.test_results)

    llm_cfg = json.load(open(args.llm_file))
    order = sorted(range(len(LLMS)), key=lambda j: llm_cfg[LLMS[j]]["input_price"])  # cheapest first
    llm_emb = np.array([llm_cfg[LLMS[j]]["embedding"] for j in order], dtype=np.float32)

    from router_planner.shared.utils import EmbeddingModel
    encoder = EmbeddingModel()
    tasks = sorted(set(tr_data["task_name"]) | set(te_data["task_name"]))
    task_emb = {t: encoder.get_embedding(TASK_DESCRIPTIONS[t]).numpy() for t in tasks}
    del encoder

    os.makedirs(args.ckpt_dir, exist_ok=True)
    results = {"config": {**OFFICIAL_CONFIG, "val_frac": args.val_frac, "scenario": "Performance First"},
               "seeds": {}}
    for seed in args.seeds:
        set_seed(seed)
        graphs, masks = build_graph(tr_data, tr_emb, tr_u[:, order], tr_c[:, order], te_data, te_emb,
                                    te_u[:, order], te_c[:, order], task_emb, llm_emb, args.val_frac, seed)
        config = {**OFFICIAL_CONFIG, "llm_num": len(LLMS),
                  "model_path": os.path.join(args.ckpt_dir, f"graphrouter_seed{seed}.pth")}
        gnn = GNN_prediction(query_feature_dim=tr_emb.shape[1], llm_feature_dim=llm_emb.shape[1],
                             hidden_features_size=config["embedding_dim"], in_edges_size=config["edge_dim"],
                             wandb=_NoWandb(), config=config, device="cpu")
        gnn.train_validate(data=graphs[0], data_validate=graphs[1], data_for_test=graphs[2])
        gnn.model.load_state_dict(torch.load(config["model_path"]))
        choice = np.array(order)[route(gnn, graphs[2], masks, len(LLMS))]  # back to LLMS indices
        results["seeds"][seed] = summarize(f"GraphRouter seed {seed}", te_data, te_u, te_c, choice)

    accs = [r["test_acc"] for r in results["seeds"].values()]
    costs = [r["test_avg_cost"] for r in results["seeds"].values()]
    results["mean"] = {k: float(np.mean([r[k] for r in results["seeds"].values()]))
                       for k in next(iter(results["seeds"].values()))}
    results["std_acc"], results["std_cost"] = float(np.std(accs)), float(np.std(costs))
    print(f"GraphRouter over {len(accs)} seeds: Acc {np.mean(accs):.3f} +- {np.std(accs):.3f}, "
          f"cost {np.mean(costs):.1f}")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
