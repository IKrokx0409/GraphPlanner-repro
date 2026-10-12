"""
Build the LLM config used for the reproduction.

The original config points at NVIDIA NIM model IDs, most of which have been
retired. This script writes a new config with currently-hosted replacements
(see API对照表.md), re-embedding each description with the same Longformer
mean-pooling used to produce the original embeddings.

Usage:
    python -m repro.build_config [--out config/llm_repro_phase2.json]
"""

import argparse
import json

from router_planner.shared.utils import EmbeddingModel

# key -> (NIM model id, price per 1M tokens, description)
PHASE2_LLMS = {
    "mistral-nemo-minitron-8b": (
        "nvidia/mistral-nemo-minitron-8b-8k-instruct", 0.2,
        "Mistral-NeMo-Minitron-8B-Instruct is an 8-billion parameter model pruned and distilled by NVIDIA "
        "from Mistral-NeMo-12B, offering strong general instruction following, reasoning and question answering "
        "for its size. It is priced at $0.20 per million input tokens and $0.20 per million output tokens.",
    ),
    "codegemma-7b": (
        "google/codegemma-7b", 0.2,
        "CodeGemma-7B is a specialized variant of the Gemma model family that focuses exclusively on code "
        "generation and completion tasks. This programming-oriented model provides robust coding assistance "
        "capabilities at an affordable rate of $0.20 per million input tokens and $0.20 per million output tokens.",
    ),
    "gemma-3-12b-it": (
        "google/gemma-3-12b-it", 0.3,
        "Gemma-3-12B-IT is a 12-billion parameter instruction-tuned model from Google's third-generation Gemma "
        "family, with improved reasoning, math and multilingual abilities over Gemma-2. It is priced at $0.30 per "
        "million input tokens and $0.30 per million output tokens.",
    ),
    "granite-3.0-8b-instruct": (
        "ibm/granite-3.0-8b-instruct", 0.2,
        "Granite-3.0-8B-Instruct is IBM's 8-billion parameter enterprise model tuned for instruction following, "
        "question answering, summarization and retrieval-augmented generation. It is priced at $0.20 per million "
        "input tokens and $0.20 per million output tokens.",
    ),
    "mistral-7b-instruct-v0.3": (
        "mistralai/mistral-7b-instruct-v0.3", 0.2,
        "Mistral-7B-Instruct-v0.3 is a fast and efficient 7-billion parameter model specifically designed for "
        "instruction-following tasks. This streamlined model provides quick response times and reliable "
        "performance at cost-effective pricing of $0.20 per million input and output tokens.",
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="config/llm_repro_phase2.json")
    args = parser.parse_args()

    model = EmbeddingModel()
    config = {}
    for key, (model_id, price, feature) in PHASE2_LLMS.items():
        config[key] = {
            "feature": feature,
            "input_price": price,
            "output_price": price,
            "model": model_id,
            "embedding": model.get_embedding(feature).tolist(),
        }
        print(f"embedded {key} -> {model_id}")

    with open(args.out, "w") as f:
        json.dump(config, f, indent=4)
    print(f"wrote {len(config)} LLMs to {args.out}")


if __name__ == "__main__":
    main()
