"""
Evaluation metrics for Task 1: Cipher Decryption.
- Character-level accuracy
- Word-level accuracy
- Levenshtein distance
"""

from typing import List
import json
import os


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def char_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Fraction of characters predicted correctly (position-wise).
    Strings are compared up to the length of the shorter one,
    remaining characters in the longer string count as errors.
    """
    total = correct = 0
    for pred, tgt in zip(predictions, targets):
        max_len = max(len(pred), len(tgt))
        for i in range(max_len):
            p = pred[i] if i < len(pred) else ""
            t = tgt[i] if i < len(tgt) else ""
            correct += int(p == t)
            total += 1
    return correct / total if total > 0 else 0.0


def word_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Fraction of whole sequences (lines) that match exactly."""
    if not predictions:
        return 0.0
    return sum(p == t for p, t in zip(predictions, targets)) / len(predictions)


def avg_levenshtein(predictions: List[str], targets: List[str]) -> float:
    """Mean Levenshtein distance over all pairs."""
    if not predictions:
        return 0.0
    return sum(levenshtein_distance(p, t) for p, t in zip(predictions, targets)) / len(predictions)


def compute_metrics(predictions: List[str], targets: List[str]) -> dict:
    """Return a dict with all three metrics."""
    return {
        "char_accuracy": char_accuracy(predictions, targets),
        "word_accuracy": word_accuracy(predictions, targets),
        "avg_levenshtein": avg_levenshtein(predictions, targets),
    }


def save_metrics(metrics: dict, output_dir: str, filename: str = "metrics.json") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path