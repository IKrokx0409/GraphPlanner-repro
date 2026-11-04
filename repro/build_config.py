"""
Build the LLM config used for the reproduction.

The original config points at NVIDIA NIM model IDs, most of which have been
retired or are not callable from a free account. This script writes a new
config whose models are served by DeepInfra:

- llama-3.1-8b-instruct, mistral-nemo-12b-instruct: models from the paper's
  pool (Table 7). Their description and embedding are copied unchanged from
  the official config; only the model id is replaced.
- llama-3.3-70b-instruct: stands in for the paper's medium models
  (Nemotron-Super-49B, Nemotron-51B, ChatQA-1.5-70B), which are all derived
  from Llama-3.x-70B.
- mistral-small-24b-instruct: stands in for Mixtral-8x7B, which DeepInfra no
  longer serves. Mistral's docs name mistral-small-latest (Small 3, 2501, at
  the time) as Mixtral-8x7B's replacement when it was retired in 2025-03.
  Llama-4-Scout was tried first but DeepInfra kept rejecting it with
  "engine_overloaded" (one call waited ~2 h), which would stall training.

New descriptions follow the official ones and are embedded with the same
Longformer mean pooling. input_price/output_price are the nominal prices of
Table 10 by model size (the router's cost signal); billing_*_price are what
DeepInfra charges per 1M tokens (checked 2026-11-02) and only feed the spend
ledger in shared/utils.py.

Usage:
    python -m repro.build_config [--out config/llm_repro_phase2.json]
"""

import argparse
import json

from router_planner.shared.utils import EmbeddingModel

OFFICIAL_CONFIG = "config/llm_descriptions_with_embeddings.json"

# key -> (DeepInfra model id, nominal price, (billing input, billing output),
#         description or None to reuse the official entry of the same key)
PHASE2_LLMS = {
    "llama-3.1-8b-instruct": (
        # FP8 ("Turbo") is the only Llama-3.1-8B variant DeepInfra serves
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", 0.2, (0.02, 0.04), None,
    ),
    "mistral-nemo-12b-instruct": (
        "mistralai/Mistral-Nemo-Instruct-2407", 0.3, (0.029, 0.03), None,
    ),
    "llama-3.3-70b-instruct": (
        # FP8 ("Turbo") is the only Llama-3.3-70B variant DeepInfra serves
        "meta-llama/Llama-3.3-70B-Instruct-Turbo", 0.9, (0.10, 0.32),
        "Llama-3.3-70B-Instruct is Meta's 70-billion parameter instruction-tuned model from the Llama-3 series, "
        "providing strong reasoning and broad world knowledge for complex and knowledge-intensive tasks, "
        "available at $0.90 per million input and output tokens.",
    ),
    "mistral-small-24b-instruct": (
        "mistralai/Mistral-Small-24B-Instruct-2501", 0.6, (0.05, 0.08),
        "Mistral-Small-24B-Instruct-2501 is a 24-billion parameter instruction-tuned model from Mistral AI, "
        "the successor to Mixtral-8x7B, offering fast responses with solid reasoning and instruction following "
        "for general-purpose tasks, available at $0.60 per million input and output tokens.",
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="config/llm_repro_phase2.json")
    args = parser.parse_args()

    official = json.load(open(OFFICIAL_CONFIG))
    model = None
    config = {}
    for key, (model_id, price, (bill_in, bill_out), feature) in PHASE2_LLMS.items():
        if feature is None:
            entry = official[key]
            assert entry["input_price"] == price, f"{key}: nominal price differs from the official config"
            feature, embedding = entry["feature"], entry["embedding"]
            source = "official"
        else:
            model = model or EmbeddingModel()
            embedding = model.get_embedding(feature).tolist()
            source = "embedded"
        config[key] = {
            "feature": feature,
            "input_price": price,
            "output_price": price,
            "model": model_id,
            "embedding": embedding,
            "billing_input_price": bill_in,
            "billing_output_price": bill_out,
        }
        print(f"{source:8s} {key} -> {model_id}")

    with open(args.out, "w") as f:
        json.dump(config, f, indent=4)
    print(f"wrote {len(config)} LLMs to {args.out}")


if __name__ == "__main__":
    main()
