"""
Router-MLP (Shnitzer et al., 2023) and Router-SVM (Hu et al., 2024) on the single-call labels.

Same setting as Router-KNN in repro/router_baselines.py: the training label of a query
is the LLM with the highest utility (ties broken by the lower cost, "Performance
First"), the router is a classifier over those labels, and its test result is the
single-call result of the LLM it picks.

Features: the Longformer query embedding GraphPlanner uses, concatenated with a
one-hot of the task (the paper describes both routers as using query features and
task information), standardised on the training queries.

Hyperparameters are chosen by 5-fold cross-validation on the training queries, by
routed accuracy (the utility of the chosen LLM), as k is for Router-KNN:
- MLP: hidden layer sizes and L2 penalty (sklearn MLPClassifier, Adam, early stopping);
  3 seeds, mean +- std.
- SVM: RBF-kernel SVC, C and gamma. Deterministic, so one run.

Usage:
    python -m repro.router_mlp_svm [--out results/router_mlp_svm.json]
"""

import argparse
import itertools
import json
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from repro.router_baselines import LLMS, best_llm, load_labels, summarize

MLP_GRID = {"hidden_layer_sizes": [(64,), (256,), (256, 64)], "alpha": [1e-4, 1e-2, 1.0]}
SVM_GRID = {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 1e-3, 1e-4]}


def features(data, emb, tasks):
    onehot = np.stack([[t == task for task in tasks] for t in data["task_name"]]).astype(np.float32)
    return np.concatenate([emb, onehot], axis=1)


def make(kind, params, seed):
    if kind == "mlp":
        return MLPClassifier(max_iter=500, early_stopping=True, random_state=seed, **params)
    return SVC(kernel="rbf", **params)


def fit_predict(kind, params, seed, x_tr, y_tr, x_te):
    scaler = StandardScaler().fit(x_tr)
    model = make(kind, params, seed).fit(scaler.transform(x_tr), y_tr)
    return model.predict(scaler.transform(x_te))


def select(kind, grid, x, y, utility, seed=0, folds=5):
    idx = np.random.RandomState(seed).permutation(len(x))
    scores = {}
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        acc = []
        for f in range(folds):
            val = idx[f::folds]
            tr = np.setdiff1d(idx, val)
            pred = fit_predict(kind, params, seed, x[tr], y[tr], x[val])
            acc.append(utility[val, pred].mean())
        scores[json.dumps(params)] = float(np.mean(acc))
    best = max(scores, key=scores.get)
    return json.loads(best), scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/router_labels_train1000.csv")
    parser.add_argument("--train_results", default="results/labels_train1000_{llm}.json")
    parser.add_argument("--test_csv", default="data/router_data_test.csv")
    parser.add_argument("--test_results", default="results/fixed_{llm}.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default="results/router_mlp_svm.json")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    tr_data, tr_emb, tr_u, tr_c = load_labels(args.train_csv, args.train_results)
    te_data, te_emb, te_u, te_c = load_labels(args.test_csv, args.test_results)
    tasks = sorted(set(tr_data["task_name"]) | set(te_data["task_name"]))
    x_tr, x_te = features(tr_data, tr_emb, tasks), features(te_data, te_emb, tasks)
    y_tr = best_llm(tr_u, tr_c)
    results = {}

    svm_params, svm_cv = select("svm", SVM_GRID, x_tr, y_tr, tr_u)
    print("Router-SVM 5-fold CV:", {k: round(v, 3) for k, v in svm_cv.items()}, "->", svm_params)
    choice = fit_predict("svm", svm_params, 0, x_tr, y_tr, x_te)
    results["router_svm"] = summarize("Router-SVM", te_data, te_u, te_c, choice)
    results["router_svm"].update(params=svm_params, cv=svm_cv)

    mlp_params, mlp_cv = select("mlp", MLP_GRID, x_tr, y_tr, tr_u)
    mlp_params["hidden_layer_sizes"] = tuple(mlp_params["hidden_layer_sizes"])  # JSON turned it into a list
    print("Router-MLP 5-fold CV:", {k: round(v, 3) for k, v in mlp_cv.items()}, "->", mlp_params)
    seeds = {}
    for seed in args.seeds:
        choice = fit_predict("mlp", mlp_params, seed, x_tr, y_tr, x_te)
        seeds[str(seed)] = summarize(f"Router-MLP seed {seed}", te_data, te_u, te_c, choice)
    accs, costs = [r["test_acc"] for r in seeds.values()], [r["test_avg_cost"] for r in seeds.values()]
    keys = [k for k in next(iter(seeds.values())) if k.startswith("test_")]
    results["router_mlp"] = {"params": {**mlp_params, "hidden_layer_sizes": list(mlp_params["hidden_layer_sizes"])},
                             "cv": mlp_cv, "seeds": seeds,
                             "mean": {k: float(np.mean([r[k] for r in seeds.values()])) for k in keys},
                             "std_acc": float(np.std(accs)), "std_cost": float(np.std(costs))}
    print(f"Router-MLP over {len(accs)} seeds: Acc {np.mean(accs):.3f} +- {np.std(accs):.3f}, cost {np.mean(costs):.1f}")
    results["train_label_share"] = {llm: float((y_tr == j).mean()) for j, llm in enumerate(LLMS)}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
