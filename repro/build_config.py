"""
Build the LLM config used for the reproduction.

The original config points at NVIDIA NIM model IDs, most of which have been
retired. This script writes a new config with currently callable replacements,
re-embedding each description with the same Longformer mean-pooling used to
produce the original embeddings.

Usage:
    python -m repro.build_config [--out config/llm_repro_phase2.json]
"""

import argparse
import json

from router_planner.shared.utils import EmbeddingModel

# key -> (NIM model id, price per 1M tokens, description, extra request options)
#
# Only these models were callable from a free NIM account on 2026-10-30; most
# other catalog entries (incl. the first replacement set) returned 404 for the
# account. Prices are nominal cost signals scaled by model size following the
# paper's Table 10 (the NIM free tier itself is not billed).
PHASE2_LLMS = {
    "nemotron-3.5-bolt-e4b": (
        "nvidia/nemotron-3.5-bolt-e4b", 0.1,
        "Nemotron-3.5-Bolt-E4B is a compact, low-latency model from NVIDIA's Nemotron 3.5 family, suited to "
        "quick instruction following and short answers at a very low cost of $0.10 per million input and "
        "output tokens.",
        {},
    ),
    "nemotron-3.5-lightning-30b-a3b": (
        "nvidia/nemotron-3.5-lightning-30b-a3b", 0.3,
        "Nemotron-3.5-Lightning-30B-A3B is a 30-billion parameter mixture-of-experts model from NVIDIA with about "
        "3 billion active parameters, offering solid general reasoning and question answering at $0.30 per "
        "million input and output tokens.",
        # Hybrid reasoning model: disable the thinking trace so answers fit the token budget
        {"extra_body": {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}}},
    ),
    "llama-3.2-90b-instruct": (
        "meta/llama-3.2-90b-vision-instruct", 1.2,
        "Llama-3.2-90B-Instruct is Meta's 90-billion parameter model (used here in text-only mode), the largest "
        "and most capable model in the pool for knowledge-intensive and complex tasks, priced at $1.20 per "
        "million input and output tokens.",
        {},
    ),
    "minimax-m3.1-flash": (
        "minimaxai/minimax-m3.1-flash", 0.6,
        "MiniMax-M3.1-Flash is a fast reasoning-oriented model from MiniMax that thinks briefly before answering, "
        "which helps on math and multi-step problems, priced at $0.60 per million input and output tokens.",
        # Its reasoning trace cannot be turned off and counts towards max_tokens
        {"max_tokens": 2048},
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="config/llm_repro_phase2.json")
    args = parser.parse_args()

    model = EmbeddingModel()
    config = {}
    for key, (model_id, price, feature, options) in PHASE2_LLMS.items():
        config[key] = {
            "feature": feature,
            "input_price": price,
            "output_price": price,
            "model": model_id,
            "embedding": model.get_embedding(feature).tolist(),
            **options,
        }
        print(f"embedded {key} -> {model_id}")

    with open(args.out, "w") as f:
        json.dump(config, f, indent=4)
    print(f"wrote {len(config)} LLMs to {args.out}")


if __name__ == "__main__":
    main()
