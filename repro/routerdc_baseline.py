"""
RouterDC baseline (Chen et al., NeurIPS 2024) on the single-call labels.

Re-implemented after the official code (github.com/shuhao02/RouterDC, commit 168c026,
train_router_mdeberta.py and train_scripts/router_train.sh). The repository has no
license, so its code is not vendored; the parts used here are written out below with
the same logic and hyperparameters:

- Router: mdeberta-v3-base encoder, first-token hidden state, cosine similarity with
  one learnable embedding per LLM (init N(0, 0.78)), temperature 1. Prediction is the
  LLM with the highest similarity. As in the official code the encoder stays in the
  mode from_pretrained returns (eval, so no dropout), and queries are truncated to
  512 tokens from the left.
- Sample-LLM contrastive loss with top_k = 3 positives and last_k = 3 negatives;
  positives that scored 0 and negatives that scored > 0.5 are masked, as in the
  official compute_sample_llm_loss.
- Sample-sample contrastive loss in the official training script's configuration:
  the cluster variant (cluster_loss_weight 1, sample_loss_weight 0, H = 3), with
  cluster ids from k-means (8 clusters) on a 5-dimensional t-SNE of the pretrained
  encoder's embeddings of the training queries (src/cluster_generate.ipynb).
- AdamW, learning rate 5e-5, batch size 64.

Adaptations to our setting:

- Scores: the utility (0/1) of each LLM's single call on the query, from the same
  labels as Router-KNN and GraphRouter.
- Training length: the official run is 1000 steps over 5 x 1000 samples (12.8
  epochs). Our training split has about 900 queries, so we keep the number of epochs
  (12.8) rather than the number of steps, which gives 180 steps (a step takes about
  50 s on our CPU).
- Memory: a batch of 64 queries at 512 tokens does not fit in 15 GB of RAM on CPU.
  Each step first encodes the batch without gradients, computes the loss and its
  gradient with respect to the 64 embeddings, then re-encodes the queries in
  micro-batches of 8 (sorted by length, padded to the longest in the micro-batch)
  and backpropagates that gradient ("gradient cache"). This gives the same loss and
  gradient as one batch of 64, because the encoder is deterministic (no dropout) and
  padding is masked out.
- Model selection: the official script keeps the checkpoint with the best accuracy
  on its training data. We hold out 10% of the training queries per task (as for
  GraphRouter) and keep the step with the best routed accuracy on them, evaluated
  every 10 steps. The test routing decisions of that step are reported.

Scenario "Performance First", 3 seeds, no API calls.

Usage:
    python -m repro.routerdc_baseline [--seeds 0 1 2] [--out results/routerdc.json]
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, DebertaV2Model

from repro.router_baselines import LLMS, load_labels, summarize

ENCODER = "microsoft/mdeberta-v3-base"
OFFICIAL_CONFIG = {  # train_scripts/router_train.sh and train_router_mdeberta.py defaults
    "top_k": 3, "last_k": 3, "learning_rate": 5e-5, "temperature": 1, "similarity": "cos",
    "sample_loss_weight": 0, "cluster_loss_weight": 1, "H": 3, "batch_size": 64,
    "max_length": 512, "n_clusters": 8, "tsne_dim": 5, "official_steps": 1000, "official_samples": 5000,
}
MICRO_BATCH = 8


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RouterDC(nn.Module):
    def __init__(self, n_llms, hidden_dim=768):
        super().__init__()
        self.backbone = DebertaV2Model.from_pretrained(ENCODER)  # eval mode, as in the official code
        self.embeddings = nn.Embedding(n_llms, hidden_dim)
        with torch.no_grad():
            nn.init.normal_(self.embeddings.weight, mean=0, std=0.78)

    def encode(self, batch):
        return self.backbone(**batch)["last_hidden_state"][:, 0, :]

    def similarity(self, hidden, t=1):
        return cosine(hidden, self.embeddings.weight) / t


def cosine(a, b):
    return (a @ b.T) / (torch.norm(a, dim=1).unsqueeze(1) * torch.norm(b, dim=1).unsqueeze(0))


def sample_llm_loss(x, scores, top_k, last_k):
    """Official compute_sample_llm_loss."""
    loss = 0
    top_true, top_index = scores.sort(dim=-1, descending=True)
    last_true, neg_index = scores.topk(k=last_k, largest=False, dim=-1)
    for i in range(top_k):
        pos_index = top_index[:, i].view(-1, 1)
        mask = torch.where(top_true[:, i].view(-1, 1) > 0, 1, 0)  # skip positives that scored 0
        top_x = torch.gather(x, 1, pos_index)
        last_x = torch.gather(x, 1, neg_index)
        last_x = torch.where(last_true > 0.5, float("-inf"), last_x)  # negatives that scored well are ignored
        log_x = torch.log(torch.softmax(torch.cat([top_x, last_x], dim=-1), dim=-1)[:, 0])
        loss += torch.mean(-log_x * mask)
    return loss


def cluster_loss(hidden, cluster_ids, H):
    """Official compute_cluster_loss: one positive from the same cluster, H negatives from others."""
    sim = cosine(hidden, hidden)
    all_index = []
    for cid in cluster_ids:
        pos = torch.nonzero(cluster_ids == cid)
        neg = torch.nonzero(cluster_ids != cid)
        if len(neg) < H:
            continue
        sel_pos = random.choice(pos)
        sel_neg = neg[random.sample(range(len(neg)), H)].view(-1)
        all_index.append(torch.cat([sel_pos, sel_neg]))
    if not all_index:  # a small last batch from a single cluster
        return hidden.sum() * 0
    rearranged = torch.gather(sim, 1, torch.stack(all_index))
    return torch.mean(-torch.log(torch.softmax(rearranged, dim=-1))[:, 0])


class Encoder:
    """Tokenises and encodes queries in length-sorted micro-batches."""

    def __init__(self, model, queries, max_length):
        tok = AutoTokenizer.from_pretrained(ENCODER, truncation_side="left")
        self.model, self.tok, self.queries, self.max_length = model, tok, list(queries), max_length
        self.length = np.array([len(tok(q, truncation=True, max_length=max_length)["input_ids"]) for q in queries])

    def chunks(self, idx, size):
        order = np.asarray(idx)[np.argsort(self.length[idx], kind="stable")]
        for j in range(0, len(order), size):
            yield order[j:j + size]

    def encode(self, idx):
        batch = self.tok([self.queries[i] for i in idx], max_length=self.max_length, truncation=True,
                         padding="longest", return_tensors="pt")
        return self.model.encode(batch)

    @torch.no_grad()
    def embed(self, idx, size=16):
        """Embeddings of the queries idx, in the order of idx."""
        out = torch.zeros(len(idx), self.model.embeddings.weight.shape[1])
        pos = {q: p for p, q in enumerate(idx)}
        for c in self.chunks(idx, size):
            out[[pos[q] for q in c]] = self.encode(c)
        return out


def make_clusters(enc, idx, seed, cfg):
    """Cluster ids of the training queries (official cluster_generate.ipynb: t-SNE to 5 dims, k-means 8)."""
    hidden = enc.embed(idx).numpy()
    low = TSNE(n_components=cfg["tsne_dim"], method="exact", random_state=seed).fit_transform(hidden)
    return KMeans(n_clusters=cfg["n_clusters"], max_iter=1000, n_init=10, random_state=42).fit_predict(low)


def train_step(model, enc, opt, idx, scores, clusters, cfg):
    """One step on batch idx with the gradient cache (same loss and gradient as one batch)."""
    pos = {q: p for p, q in enumerate(idx)}
    hidden = enc.embed(idx, size=MICRO_BATCH).requires_grad_()
    x = model.similarity(hidden, cfg["temperature"])
    loss = sample_llm_loss(x, scores, cfg["top_k"], cfg["last_k"])
    if cfg["cluster_loss_weight"]:
        loss = loss + cfg["cluster_loss_weight"] * cluster_loss(hidden, clusters, cfg["H"])
    opt.zero_grad()
    loss.backward()  # gradients for the LLM embeddings, and d loss / d hidden
    grad = hidden.grad
    for c in enc.chunks(idx, MICRO_BATCH):
        enc.encode(c).backward(grad[[pos[q] for q in c]])
    opt.step()
    return loss.item()


@torch.no_grad()
def route(model, enc, idx):
    return torch.argmax(model.similarity(enc.embed(idx)), dim=1).numpy()


def run_seed(seed, tr_data, tr_u, te_n, enc, cfg, steps, eval_every, val_frac, resume_path, log):
    n_tr = len(tr_data)
    rng = np.random.RandomState(seed)
    val = []
    for t in sorted(tr_data["task_name"].unique()):
        q = np.flatnonzero(tr_data["task_name"].values == t)
        val.extend(rng.choice(q, max(1, int(round(len(q) * val_frac))), replace=False))
    val = np.array(sorted(val))
    train = np.setdiff1d(np.arange(n_tr), val)
    test = np.arange(n_tr, n_tr + te_n)

    set_seed(seed)
    model = RouterDC(len(LLMS))
    enc.model = model
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    state = {"step": 0, "epoch": 0, "best_val": -1.0, "best_step": None, "best_test_choice": None, "log": []}

    if os.path.exists(resume_path):
        ck = torch.load(resume_path, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        random.setstate(ck["py_rng"])
        np.random.set_state(ck["np_rng"])
        torch.set_rng_state(ck["torch_rng"])
        clusters = ck["clusters"]
        log(f"seed {seed}: resumed at step {state['step']}")
    else:
        clusters = make_clusters(enc, train, seed, cfg)
        log(f"seed {seed}: {len(train)} train / {len(val)} val queries, cluster sizes {np.bincount(clusters).tolist()}")

    scores_all = torch.tensor(tr_u, dtype=torch.float32)
    cluster_of = dict(zip(train, clusters))
    bs = cfg["batch_size"]
    while state["step"] < steps:
        # Batches of an epoch come from a permutation fixed by (seed, epoch), so a resumed
        # run sees the same batches; the last batch of an epoch is smaller (no drop_last).
        perm = np.random.RandomState(seed * 1000 + state["epoch"]).permutation(train)
        start = state.get("batch_in_epoch", 0) * bs
        for b in range(start, len(perm), bs):
            idx = perm[b:b + bs]
            t0 = time.time()
            loss = train_step(model, enc, opt, idx, scores_all[idx],
                              torch.tensor([cluster_of[i] for i in idx]), cfg)
            state["step"] += 1
            state["batch_in_epoch"] = b // bs + 1
            entry = {"step": state["step"], "loss": loss, "sec": time.time() - t0}
            if state["step"] % eval_every == 0 or state["step"] == steps:
                choice = route(model, enc, val)
                entry["val_acc"] = float(tr_u[val, choice].mean())
                if entry["val_acc"] > state["best_val"]:
                    state["best_val"], state["best_step"] = entry["val_acc"], state["step"]
                    state["best_test_choice"] = route(model, enc, test)
            state["log"].append(entry)
            if "val_acc" in entry:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                            "py_rng": random.getstate(), "np_rng": np.random.get_state(),
                            "torch_rng": torch.get_rng_state(), "clusters": clusters}, resume_path)
            log(f"seed {seed} step {state['step']}/{steps} epoch {state['epoch']} loss {loss:.4f} "
                f"({entry['sec']:.0f}s)" + (f" val_acc {entry['val_acc']:.3f} best {state['best_val']:.3f}"
                                             f"@{state['best_step']}" if "val_acc" in entry else ""))
            if state["step"] >= steps:
                break
        else:
            state["epoch"] += 1
            state["batch_in_epoch"] = 0
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/router_labels_train1000.csv")
    parser.add_argument("--train_results", default="results/labels_train1000_{llm}.json")
    parser.add_argument("--test_csv", default="data/router_data_test.csv")
    parser.add_argument("--test_results", default="results/fixed_{llm}.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=None, help="default: the official number of epochs")
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--ckpt_dir", default="checkpoints_phase2/routerdc")
    parser.add_argument("--out", default="results/routerdc.json")
    args = parser.parse_args()
    cfg = dict(OFFICIAL_CONFIG)

    tr_data, _, tr_u, tr_c = load_labels(args.train_csv, args.train_results)
    te_data, _, te_u, te_c = load_labels(args.test_csv, args.test_results)
    n_train = len(tr_data) - sum(max(1, int(round(n * args.val_frac)))
                                 for n in tr_data["task_name"].value_counts().values)
    epochs = cfg["official_steps"] * cfg["batch_size"] / cfg["official_samples"]
    steps = args.steps or int(round(epochs * n_train / cfg["batch_size"]))
    cfg.update(steps=steps, epochs=epochs, eval_every=args.eval_every, val_frac=args.val_frac,
               micro_batch=MICRO_BATCH, scenario="Performance First")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    def log(msg):
        print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)

    log(f"RouterDC: {n_train} training queries, {epochs:.1f} epochs = {steps} steps of {cfg['batch_size']}")
    queries = list(tr_data["query"]) + list(te_data["query"])
    results = {"config": cfg, "seeds": {}}
    if os.path.exists(args.out):  # seeds finished in an earlier run
        results["seeds"] = json.load(open(args.out))["seeds"]
    for seed in args.seeds:
        if str(seed) in results["seeds"]:
            log(f"seed {seed}: already in {args.out}, skipped")
            continue
        enc = Encoder(None, queries, cfg["max_length"])
        state = run_seed(seed, tr_data, tr_u, len(te_data), enc, cfg, steps, args.eval_every, args.val_frac,
                         os.path.join(args.ckpt_dir, f"routerdc_seed{seed}_resume.pt"), log)
        choice = np.asarray(state["best_test_choice"])
        stats = summarize(f"RouterDC seed {seed}", te_data, te_u, te_c, choice)
        stats.update(best_step=state["best_step"], best_val_acc=state["best_val"], train_log=state["log"])
        results["seeds"][str(seed)] = stats
        with open(args.out, "w") as f:  # after every seed, so a crash keeps the finished ones
            json.dump(results, f, indent=2)
        del enc, state

    runs = [results["seeds"][str(s)] for s in args.seeds]
    accs, costs = [r["test_acc"] for r in runs], [r["test_avg_cost"] for r in runs]
    keys = [k for k in runs[0] if k.startswith("test_")]
    results["mean"] = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    results["std_acc"], results["std_cost"] = float(np.std(accs)), float(np.std(costs))
    log(f"RouterDC over {len(accs)} seeds: Acc {np.mean(accs):.3f} +- {np.std(accs):.3f}, cost {np.mean(costs):.1f}")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
