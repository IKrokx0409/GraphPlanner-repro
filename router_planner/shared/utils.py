"""
Shared utility functions for the RouterPlanner framework.

Provides:
- EmbeddingModel: Longformer-based text embedding with mean pooling
- File I/O helpers for JSON and pickle formats
- Text evaluation metrics (F1, EM, CEM, BERT score, CommonGen coverage)
- LLM API client with thread-safe round-robin key rotation
"""

import os
import json
import re
import string
import pickle
from collections import Counter
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import LongformerModel, LongformerTokenizer
from sklearn.decomposition import PCA
from bert_score import score as bert_score_fn
import openai
from openai import OpenAI
import itertools
import random
import threading
import time

from .api_guard import BudgetExceeded, ResponseCache, SpendLedger


# ---------------------------------------------------------------------------
# LLM API Client Setup
# ---------------------------------------------------------------------------
# The reproduction calls the models through DeepInfra (OpenAI-compatible).
# Any other OpenAI-compatible endpoint can be used via LLM_BASE_URL plus
# LLM_API_KEYS (comma-separated keys, rotated round-robin), e.g. NVIDIA NIM:
#   export LLM_BASE_URL="https://integrate.api.nvidia.com/v1" LLM_API_KEYS="nvapi-..."

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai")
_api_keys_str = os.environ.get("LLM_API_KEYS") or os.environ.get("DEEPINFRA_API_KEY", "")
API_KEYS = [k.strip() for k in _api_keys_str.split(",") if k.strip()]

# Set MOCK_LLM=1 to replace API calls with canned responses (offline smoke tests).
MOCK_LLM = os.environ.get("MOCK_LLM", "0") == "1"

if not API_KEYS and not MOCK_LLM:
    import warnings
    warnings.warn(
        "No API key found (DEEPINFRA_API_KEY or LLM_API_KEYS). LLM API calls will fail."
    )

# Fail fast on a hung request; model_prompting does the retrying (see there).
clients = [
    OpenAI(
        base_url=LLM_BASE_URL,
        api_key=key,
        timeout=60,
        max_retries=0,
    )
    for key in API_KEYS
]

# Thread-safe round-robin iterator for load balancing across API keys
_client_cycle = itertools.cycle(clients) if clients else itertools.cycle([None])
_client_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------
class EmbeddingModel:
    """
    Text embedding model based on Longformer with mean pooling.

    Uses the AllenAI Longformer (supports up to 4096 tokens) to produce
    fixed-size embeddings by averaging the last hidden state over
    non-padding positions.
    """

    def __init__(self, model_name: str = "allenai/longformer-base-4096", device: str = "cpu"):
        self.device = device
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.model = LongformerModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def get_embedding(self, text: str) -> torch.Tensor:
        """
        Compute a single embedding vector for the given text.

        Args:
            text: Input text string.

        Returns:
            1-D tensor of shape [hidden_dim] on CPU.
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=4096,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)

            # Mean pooling over non-padding tokens
            last_hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = summed / counts

        return mean_pooled.squeeze(0).cpu()


# ---------------------------------------------------------------------------
# File I/O Helpers
# ---------------------------------------------------------------------------
def loadjson(filename: str) -> dict:
    """Load and return data from a JSON file."""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def savejson(data: dict, filename: str) -> None:
    """Save a dictionary to a JSON file with indentation."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def loadpkl(filename: str):
    """Load and return an object from a pickle file."""
    with open(filename, "rb") as f:
        return pickle.load(f)


def savepkl(data, filename: str) -> None:
    """Save an object to a pickle file."""
    with open(filename, "wb") as f:
        pickle.dump(data, f)


# ---------------------------------------------------------------------------
# Text Normalization & Evaluation Metrics
# ---------------------------------------------------------------------------
def normalize_answer(s: str, normal_method: str = "") -> str:
    """
    Normalize text for evaluation by lowercasing, removing articles,
    punctuation, and extra whitespace.

    Args:
        s: Input string.
        normal_method: "mc" for multiple-choice extraction, "" for standard.

    Returns:
        Normalized string.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    def mc_remove(text):
        matches = re.findall(r"\(\s*[a-zA-Z]\s*\)", text)
        return matches[-1] if matches else ""

    if normal_method == "mc":
        return mc_remove(s)
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    Compute token-level F1 score between prediction and ground truth.

    Returns:
        Tuple of (f1, precision, recall).
    """
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    ZERO = (0.0, 0.0, 0.0)

    if norm_pred in ["yes", "no", "noanswer"] and norm_pred != norm_gt:
        return ZERO
    if norm_gt in ["yes", "no", "noanswer"] and norm_pred != norm_gt:
        return ZERO

    pred_tokens = norm_pred.split()
    gt_tokens = norm_gt.split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str, normal_method: str = "") -> bool:
    """Check if prediction exactly matches ground truth after normalization."""
    if normal_method == "mc":
        return ground_truth.strip().lower() in normalize_answer(prediction, "mc").strip().lower()
    return normalize_answer(prediction, normal_method) == normalize_answer(ground_truth, normal_method)


def cem_score(prediction: str, ground_truth: str) -> float:
    """
    Contained Exact Match score.
    Returns 1.0 if the normalized ground truth is contained in the prediction.
    """
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    return 1.0 if (norm_pred == norm_gt or norm_gt in norm_pred) else 0.0


def cemf1_score(prediction: str, ground_truth: str) -> float:
    """
    CEM-F1: returns 1.0 on exact/contained match, otherwise token-level F1.
    """
    if cem_score(prediction, ground_truth) == 1.0:
        return 1.0
    return f1_score(prediction, ground_truth)[0]


def get_bert_score(generated: List[str], references: List[str]) -> float:
    """
    Compute average BERT-score F1 between generated texts and references.
    """
    scores = []
    for gen, ref in zip(generated, references):
        _, _, F = bert_score_fn([gen], [ref], lang="en", verbose=False)
        scores.append(F.mean().item())
    return np.mean(scores)


# ---------------------------------------------------------------------------
# CommonGen Evaluation
# ---------------------------------------------------------------------------
def _simple_lemmatize(word: str) -> str:
    """Simple rule-based lemmatization for concept matching."""
    word = word.lower().strip()
    for suffix in ["ing", "ed", "er", "s"]:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[: -len(suffix)]
    return word


def evaluate_commongen_coverage(prediction: str, ground_truth: str) -> float:
    """
    Evaluate concept coverage for the CommonGen task.

    Args:
        prediction: Generated text.
        ground_truth: Comma-separated concept string.

    Returns:
        Fraction of ground-truth concepts found in the prediction.
    """
    if not prediction or not ground_truth:
        return 0.0

    concepts = [c.strip() for c in ground_truth.split(",") if c.strip()]
    if not concepts:
        return 0.0

    gt_lemmas = [_simple_lemmatize(c) for c in concepts]
    pred_lemmas = [_simple_lemmatize(w) for w in prediction.split()]

    matched = sum(1 for g in gt_lemmas if g in pred_lemmas)
    return matched / len(concepts)


# ---------------------------------------------------------------------------
# Embedding Dimensionality Reduction
# ---------------------------------------------------------------------------
def reduce_embedding_dim(embeddings: np.ndarray, dim: int = 50) -> np.ndarray:
    """Reduce embedding dimensionality using PCA."""
    pca = PCA(n_components=dim)
    return pca.fit_transform(embeddings)


# ---------------------------------------------------------------------------
# LLM API Prompting
# ---------------------------------------------------------------------------
def model_prompting(
    llm_model: str,
    prompt: str,
    max_token_num: Optional[int] = 512,
    temperature: Optional[float] = 0.1,
    top_p: Optional[float] = 0.7,
) -> str:
    """
    Send a prompt to an LLM via the NVIDIA API (OpenAI-compatible).

    Uses thread-safe round-robin rotation across configured API keys
    for load balancing.

    Args:
        llm_model: Model identifier (e.g., "meta/llama-3.1-8b-instruct").
        prompt: User prompt text.
        max_token_num: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.

    Returns:
        Generated text response.
    """
    with _client_lock:
        API_STATS["calls"] += 1

    if MOCK_LLM:
        return _mock_response(prompt)

    # Per-model overrides from the LLM config (see register_model_options)
    options = MODEL_OPTIONS.get(llm_model, {})
    max_token_num = options.get("max_tokens", max_token_num)
    if llm_model not in BILLING_PRICES:
        raise ValueError(f"No billing price registered for {llm_model}; add billing_*_price to the LLM config")

    cache_key = ResponseCache.make_key(model=llm_model, prompt=prompt, max_tokens=max_token_num,
                                       temperature=temperature, top_p=top_p, extra_body=options.get("extra_body"))
    cache = _response_cache()
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            with _client_lock:
                API_STATS["cache_hits"] += 1
            return cached

    # Network problems, rate limits and server errors pause the call until the API
    # is reachable again, so an outage never turns into wrong answers. Account or
    # configuration errors stop the run. Only a request the API rejects as such
    # (e.g. 400) is retried a few times and then answered empty (scored as wrong).
    waited, rejected = 0.0, 0
    while True:
        # Outside the try below: hitting the budget must stop the run, not count as a failed call
        _spend_ledger().check()

        # Thread-safe client selection
        with _client_lock:
            client = next(_client_cycle)

        if client is None:
            raise RuntimeError("No API keys configured. Set DEEPINFRA_API_KEY or LLM_API_KEYS.")

        try:
            response = _create_with_deadline(
                client, llm_model, prompt,
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_token_num,
                temperature=temperature,
                top_p=top_p,
                stream=False,
                extra_body=options.get("extra_body"),
            )
        except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError,
                CallDeadlineExceeded) as e:
            # APIConnectionError includes timeouts
            if not API_WAIT_ON_OUTAGE:
                raise
            delay = min(60.0, 5.0 * 2 ** min(int(waited // 60), 4)) + random.uniform(0, 5)
            print(f"API unreachable for model {llm_model} ({type(e).__name__}: {e}); "
                  f"waited {waited:.0f}s so far, retrying in {delay:.0f}s")
            with _client_lock:
                API_STATS["outage_retries"] += 1
            time.sleep(delay)
            waited += delay
            continue
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError) as e:
            raise RuntimeError(f"API refused model {llm_model}: {e}") from e
        except openai.APIStatusError as e:
            if e.status_code == 402:
                raise RuntimeError(f"API account out of credit: {e}") from e
            rejected += 1
            print(f"API rejected request for model {llm_model} (attempt {rejected}): {e}")
            if rejected > API_EXTRA_RETRIES:
                break
            time.sleep(5 * rejected)
            continue

        if cache is not None and response:
            cache.put(cache_key, llm_model, response)
        return response

    with _client_lock:
        API_STATS["failures"] += 1
    return ""


class CallDeadlineExceeded(Exception):
    """An API call did not return within API_CALL_DEADLINE_S (e.g. a connection hung by a network switch)."""


def _create_with_deadline(client, llm_model: str, prompt: str, **request) -> str:
    """
    Run one chat completion in a helper thread and give up after API_CALL_DEADLINE_S.

    The HTTP client's own timeout does not fire on every kind of stalled
    connection. An abandoned call keeps running in its daemon thread; if it
    completes later, it is still billed to the ledger, so the spend stays exact.
    """
    box = {}

    def run():
        try:
            completion = client.chat.completions.create(**request)
            response = completion.choices[0].message.content or ""
            _record_spend(llm_model, completion.usage, prompt, response)
            box["response"] = response
        except BaseException as e:
            box["error"] = e

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(API_CALL_DEADLINE_S)
    if worker.is_alive():
        raise CallDeadlineExceeded(f"no response within {API_CALL_DEADLINE_S:.0f}s")
    if "error" in box:
        raise box["error"]
    return box["response"]


def _record_spend(llm_model: str, usage, prompt: str, response: str) -> None:
    """Bill one call to the shared ledger using the token usage reported by the API."""
    if usage is not None and usage.prompt_tokens is not None:
        prompt_tokens, completion_tokens = usage.prompt_tokens, usage.completion_tokens or 0
    else:
        # Conservative fallback (~3 characters per token) if the provider omits usage
        print(f"Warning: no token usage returned for {llm_model}; estimating from text length")
        prompt_tokens, completion_tokens = len(prompt) // 3 + 1, len(response) // 3 + 1
    in_price, out_price = BILLING_PRICES[llm_model]
    usd = (prompt_tokens * in_price + completion_tokens * out_price) / 1e6
    _spend_ledger().record(llm_model, prompt_tokens, completion_tokens, usd)
    with _client_lock:
        API_STATS["prompt_tokens"] += prompt_tokens
        API_STATS["completion_tokens"] += completion_tokens
        API_STATS["usd"] += usd


# Per-model request overrides, keyed by model id: {"max_tokens": int, "extra_body": dict}.
MODEL_OPTIONS = {}

# Real per-1M-token prices billed by the provider, keyed by model id: (input, output).
# These feed the spend ledger only; the router's cost signal uses input_price/output_price.
BILLING_PRICES = {}


def register_model_options(llm_data: dict) -> None:
    """Register the optional request overrides and the billing prices of an LLM config."""
    for info in llm_data.values():
        options = {k: info[k] for k in ("max_tokens", "extra_body") if k in info}
        if options:
            MODEL_OPTIONS[info["model"]] = options
        if "billing_input_price" in info:
            BILLING_PRICES[info["model"]] = (info["billing_input_price"], info["billing_output_price"])


# Shared budget (USD) across all processes using the same ledger file.
LLM_BUDGET_USD = float(os.environ.get("LLM_BUDGET_USD", "8"))
LLM_SPEND_DB = os.environ.get("LLM_SPEND_DB", "cache/api_spend.sqlite")
# Set LLM_CACHE_DB=off to always call the API.
LLM_CACHE_DB = os.environ.get("LLM_CACHE_DB", "cache/llm_responses.sqlite")

_guard_lock = threading.Lock()
_guards = {}


def _spend_ledger() -> SpendLedger:
    with _guard_lock:
        if "ledger" not in _guards:
            _guards["ledger"] = SpendLedger(LLM_SPEND_DB, LLM_BUDGET_USD)
        return _guards["ledger"]


def _response_cache() -> Optional[ResponseCache]:
    if LLM_CACHE_DB == "off":
        return None
    with _guard_lock:
        if "cache" not in _guards:
            _guards["cache"] = ResponseCache(LLM_CACHE_DB)
        return _guards["cache"]


# Retries of a request the API rejects (e.g. 400) before answering it empty.
API_EXTRA_RETRIES = 3

# Wait for the API to come back after network errors (False: raise immediately).
API_WAIT_ON_OUTAGE = True

# Hard wall-clock limit for one API call, on top of the HTTP client's timeout.
API_CALL_DEADLINE_S = 90.0

# Running counters, logged by the training scripts.
API_STATS = {"calls": 0, "failures": 0, "cache_hits": 0, "outage_retries": 0,
             "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}


def _mock_response(prompt: str) -> str:
    """Canned responses that exercise every code path without a real LLM."""
    if "query decomposition assistant" in prompt:
        return "What is the first part of the problem?\nWhat is the second part of the problem?"
    if "professional summarizer" in prompt:
        return "Combining the child answers, the answer is 42."
    return "The answer is 42. Answer: A"
