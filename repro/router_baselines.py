"""
Offline single-round router baselines (Router-KNN; oracle and best-single references).

Single-round routers pick one LLM per query and call it once, so they can be
evaluated offline from single-call labels: every LLM answers every query once
(repro/run_phase2.sh, stage "labels" for training queries and the fixed-policy
results on the test set). A router's test result is the label of the LLM it
selects.

Training label of a query (Performance First, as in Table 3): the LLM with the
highest utility, ties broken by the lower cost.

Router-KNN (Shnitzer et al., 2023): majority label of the k nearest training
queries by cosine similarity of the query embeddings (the Longformer embeddings
GraphPlanner uses), ties broken by the cheaper LLM. k (1 to 400) is chosen by 5-fold
cross-validation on the training labels. Sensitivity variant "Router-KNN
(score)": the LLM with the highest mean utility over the k neighbours.

Usage:
    python -m repro.router_baselines [--out results/router_baselines.json]
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from router_planner.dynamic_router.route_env import Env_route
from router_planner.dynamic_router.train import TASK_DOMAINS

LLMS = ["llama-3.1-8b-instruct", "mistral-nemo-12b-instruct",
        "llama-3.3-70b-instruct", "mistral-small-24b-instruct"]
K_GRID = [1, 3, 5, 10, 20, 50, 100, 200, 400]


def load_labels(csv_path, result_pattern):
    """Return (queries DataFrame, query embeddings, utility [n, m], cost [n, m]) aligned by (task, query)."""
    data = pd.read_csv(csv_path)
    keys = list(zip(data["task_name"], data["query"]))
    utility = np.full((len(data), len(LLMS)), np.nan)
    cost = np.full((len(data), len(LLMS)), np.nan)
    for j, llm in enumerate(LLMS):
        by_key = defaultdict(list)
        for r in json.load(open(result_pattern.format(llm=llm)))["records"]:
            by_key[(r["task"], r["query"])].append(r)
        for i, key in enumerate(keys):
            r = by_key[key].pop()  # a query text can occur twice; any copy is fine
            utility[i, j], cost[i, j] = r["utility"], r["cost"]
    assert not np.isnan(utility).any(), f"missing labels for {result_pattern}"
    emb = np.stack([Env_route.to_tensor(None, e).numpy() for e in data["query_embedding"]]).astype(np.float32)
    return data, emb, utility, cost


def best_llm(utility, cost):
    """Index of the highest-utility LLM per row, ties broken by the lower cost."""
    return np.array([np.lexsort((c, -u))[0] for u, c in zip(utility, cost)])


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def knn_predict(train_emb, train_label, test_emb, k, llm_cost_rank):
    """Majority label among the k nearest neighbours; ties go to the cheaper LLM."""
    sim = normalize(test_emb) @ normalize(train_emb).T
    nearest = np.argsort(-sim, axis=1)[:, :k]
    preds = []
    for row in nearest:
        votes = np.bincount(train_label[row], minlength=len(LLMS))
        tied = np.flatnonzero(votes == votes.max())
        preds.append(tied[np.argmin(llm_cost_rank[tied])])
    return np.array(preds)


def knn_score_predict(train_emb, train_utility, test_emb, k, llm_cost_rank):
    """Sensitivity variant: LLM with the highest mean utility over the k nearest neighbours.

    With binary utilities most training queries are ties (every LLM right or every
    LLM wrong), which the majority-label rule turns into labels of the cheapest LLM;
    averaging utilities per LLM does not discard that information.
    """
    sim = normalize(test_emb) @ normalize(train_emb).T
    nearest = np.argsort(-sim, axis=1)[:, :k]
    preds = []
    for row in nearest:
        mean_u = train_utility[row].mean(axis=0)
        tied = np.flatnonzero(np.isclose(mean_u, mean_u.max()))
        preds.append(tied[np.argmin(llm_cost_rank[tied])])
    return np.array(preds)


def select_k(emb, utility, cost, label, llm_cost_rank, folds=5, seed=0, score_variant=False):
    idx = np.random.RandomState(seed).permutation(len(emb))
    scores = {}
    for k in K_GRID:
        acc = []
        for f in range(folds):
            val = idx[f::folds]
            tr = np.setdiff1d(idx, val)
            if score_variant:
                pred = knn_score_predict(emb[tr], utility[tr], emb[val], k, llm_cost_rank)
            else:
                pred = knn_predict(emb[tr], label[tr], emb[val], k, llm_cost_rank)
            acc.append(utility[val, pred].mean())
        scores[k] = float(np.mean(acc))
    return max(scores, key=scores.get), scores


def summarize(name, data, utility, cost, choice):
    u = utility[np.arange(len(choice)), choice]
    c = cost[np.arange(len(choice)), choice]
    stats = {"test_acc": float(u.mean()), "test_avg_cost": float(c.mean()), "test_avg_llm_calls": 1.0,
             "test_samples": int(len(choice))}
    domains = data["task_name"].map(TASK_DOMAINS)
    for d in sorted(domains.unique()):
        stats[f"test_acc_domain/{d}"] = float(u[domains.values == d].mean())
        stats[f"test_cost_domain/{d}"] = float(c[domains.values == d].mean())
    for j, llm in enumerate(LLMS):
        stats[f"test_llm_share/{llm}"] = float((choice == j).mean())
    line = "  ".join(f"{d} {stats[f'test_acc_domain/{d}']:.3f}" for d in sorted(domains.unique()))
    print(f"{name:22s} Acc {stats['test_acc']:.3f}  cost {stats['test_avg_cost']:6.1f}  {line}")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/router_labels_train1000.csv")
    parser.add_argument("--train_results", default="results/labels_train1000_{llm}.json")
    parser.add_argument("--test_csv", default="data/router_data_test.csv")
    parser.add_argument("--test_results", default="results/fixed_{llm}.json")
    parser.add_argument("--out", default="results/router_baselines.json")
    args = parser.parse_args()

    _, tr_emb, tr_u, tr_c = load_labels(args.train_csv, args.train_results)
    te_data, te_emb, te_u, te_c = load_labels(args.test_csv, args.test_results)
    tr_label = best_llm(tr_u, tr_c)
    llm_cost_rank = np.argsort(np.argsort(tr_c.mean(axis=0)))  # 0 = cheapest on average

    k, cv = select_k(tr_emb, tr_u, tr_c, tr_label, llm_cost_rank)
    print("Router-KNN 5-fold CV accuracy by k:", {kk: round(v, 3) for kk, v in cv.items()}, "-> k =", k)

    results = {}
    for j, llm in enumerate(LLMS):
        results[f"single/{llm}"] = summarize(f"single {llm}", te_data, te_u, te_c, np.full(len(te_data), j))
    best_single = int(np.argmax(tr_u.mean(axis=0)))  # chosen on training labels, not on test
    results["best_single_on_train"] = summarize(f"best single (train)", te_data, te_u, te_c,
                                                np.full(len(te_data), best_single))
    results["router_knn"] = summarize(f"Router-KNN (k={k})", te_data, te_u, te_c,
                                      knn_predict(tr_emb, tr_label, te_emb, k, llm_cost_rank))
    results["router_knn"]["k"], results["router_knn"]["cv"] = k, cv
    ks, cvs = select_k(tr_emb, tr_u, tr_c, tr_label, llm_cost_rank, score_variant=True)
    print("Router-KNN (score) 5-fold CV accuracy by k:", {kk: round(v, 3) for kk, v in cvs.items()}, "-> k =", ks)
    results["router_knn_score"] = summarize(f"Router-KNN score (k={ks})", te_data, te_u, te_c,
                                            knn_score_predict(tr_emb, tr_u, te_emb, ks, llm_cost_rank))
    results["router_knn_score"]["k"], results["router_knn_score"]["cv"] = ks, cvs
    results["oracle"] = summarize("oracle (upper bound)", te_data, te_u, te_c, best_llm(te_u, te_c))
    results["train_label_share"] = {llm: float((tr_label == j).mean()) for j, llm in enumerate(LLMS)}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
