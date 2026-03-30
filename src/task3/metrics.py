"""
src/task3/metrics.py

Extended evaluation metrics for Task 3.
Reuses Task 1 character/word metrics and adds BLEU and ROUGE.
"""

import os
import json
from typing import List

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Reuse Task 1 primitive metrics
from src.task1.metrics import (
    char_accuracy,
    word_accuracy,
    avg_levenshtein,
)

# Download required NLTK data once (safe to call multiple times)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def bleu_score(predictions: List[str], targets: List[str]) -> float:
    """
    Corpus-level BLEU score (1-gram through 4-gram).
    Uses NLTK SmoothingFunction method1 to handle short sequences.

    Both preds and targets are plain text strings (space-separated words).
    """
    smoothie = SmoothingFunction().method1
    references = [[tgt.split()] for tgt in targets]   # [[ref_tokens], ...]
    hypotheses = [pred.split() for pred in predictions]  # [hyp_tokens, ...]
    return corpus_bleu(references, hypotheses, smoothing_function=smoothie)


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------

def rouge_scores(predictions: List[str], targets: List[str]) -> dict:
    """
    Average ROUGE-1, ROUGE-2, and ROUGE-L F1 scores over all pairs.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    r1_total = r2_total = rL_total = 0.0
    n = len(predictions)

    from tqdm import tqdm
    for pred, tgt in tqdm(zip(predictions, targets), desc="Eval ROUGE", leave=False, total=n):
        scores = scorer.score(tgt, pred)
        r1_total += scores["rouge1"].fmeasure
        r2_total += scores["rouge2"].fmeasure
        rL_total += scores["rougeL"].fmeasure

    if n == 0:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    return {
        "rouge1": r1_total / n,
        "rouge2": r2_total / n,
        "rougeL": rL_total / n,
    }


# ---------------------------------------------------------------------------
# Perplexity helpers
# ---------------------------------------------------------------------------

def ssm_perplexity(model, sequences: "torch.Tensor", pad_idx: int, device) -> float:
    """
    Standard NWP perplexity:  exp(mean CE loss over non-PAD positions).

    sequences : LongTensor [N, 20]  (pre-chunked, padded)
    """
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_size = 256

    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size), desc="SSM Perplexity", leave=False):
            batch = sequences[start : start + batch_size].to(device)
            inp = batch[:, :-1]
            tgt = batch[:, 1:]

            logits = model(inp)                           # [B, 19, V]
            logits_flat = logits.view(-1, logits.size(-1))
            tgt_flat = tgt.reshape(-1)

            mask = tgt_flat != pad_idx
            if mask.sum() == 0:
                continue

            loss = F.cross_entropy(
                logits_flat[mask], tgt_flat[mask], reduction="sum"
            )
            total_loss   += loss.item()
            total_tokens += mask.sum().item()

    if total_tokens == 0:
        return float("inf")
    return float(torch.exp(torch.tensor(total_loss / total_tokens)))


def bilstm_perplexity(model, sequences: "torch.Tensor", mask_idx: int, pad_idx: int, device, batch_size: int = 256) -> float:
    """
    Optimized MLM perplexity — batched and randomized 15% evaluation.
    """
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    # Deterministically mask 15% of valid tokens per sequence for high-speed
    # rigorous sampling. Guaranteed reproducible.
    torch.manual_seed(42)

    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size), desc="BiLSTM Perplexity", leave=False):
            batch = sequences[start : start + batch_size].to(device)  # [B, T]
            
            targets = torch.full_like(batch, fill_value=-100)
            masked_inputs = batch.clone()
            
            valid_mask = (batch != pad_idx) & (batch != mask_idx)
            rand_matrix = torch.rand(batch.shape, device=device)
            
            # Select 15% of tokens
            mask_positions = valid_mask & (rand_matrix < 0.15)
            
            # Fallback: if a sequence drew no masks, randomly force 1
            no_mask = mask_positions.sum(dim=1) == 0
            for i in range(batch.shape[0]):
                if no_mask[i] and valid_mask[i].sum() > 0:
                    valid_indices = torch.nonzero(valid_mask[i]).squeeze(1)
                    chosen = valid_indices[torch.randint(0, len(valid_indices), (1,)).item()]
                    mask_positions[i, chosen] = True
            
            targets[mask_positions] = batch[mask_positions]
            masked_inputs[mask_positions] = mask_idx
            
            logits = model(masked_inputs)  # [B, T, V]
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-100,
                reduction="sum"
            )
            
            num_masked = mask_positions.sum().item()
            if num_masked > 0:
                total_loss += loss.item()
                total_tokens += num_masked

    if total_tokens == 0:
        return float("inf")
    return float(torch.exp(torch.tensor(total_loss / total_tokens)))


# ---------------------------------------------------------------------------
# Combined metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(predictions: List[str], targets: List[str]) -> dict:
    """Return all metrics as a flat dict."""
    return {
        "char_accuracy":    char_accuracy(predictions, targets),
        "word_accuracy":    word_accuracy(predictions, targets),
        "avg_levenshtein":  avg_levenshtein(predictions, targets),
        "bleu":             bleu_score(predictions, targets),
        **rouge_scores(predictions, targets),
    }


def save_metrics(metrics: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
