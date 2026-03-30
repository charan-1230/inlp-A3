"""
src/task3/pipeline.py

Full Task 3 pipeline:
    cipher_0x.txt
        → Task1 LSTM (character decryption)
        → noisy plain text
        → Task2 LM (SSM or BiLSTM, word-level correction)
        → corrected plain text
        → metrics vs ground truth

Entry point: main(config_path, mode)
"""

from __future__ import annotations

import os
import sys
import json
import math
import pickle
import time
from datetime import datetime
from typing import List, Tuple

import torch
from tqdm import tqdm
import torch.nn.functional as F
import yaml

from src.preprocessing.task1.tokenizer import tokenize_cipher_line
from src.preprocessing.task2.tokenizer import tokenize_line
from src.task1.dataset import load_vocab as load_task1_vocab
from src.task1.model_lstm import SeqLabelLSTM
from src.task1.trainer import indices_to_string
from src.task2.models.ssm_model import SSMModel
from src.task2.models.bilstm_model import BiLSTMModel
from src.task2.utils import load_vocab_and_sequences
from src.task3.metrics import compute_all_metrics, save_metrics, ssm_perplexity, bilstm_perplexity
from src.utils.checkpoints import load_checkpoint
from src.utils.hf_wandb import pull_from_hub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_ckpt(model: torch.nn.Module, cfg: dict, label: str, device: torch.device) -> None:
    """Load checkpoint from local path, falling back to HuggingFace Hub."""
    local_path = cfg.get("checkpoint_path", "")
    if local_path and os.path.isfile(local_path):
        print(f"[{label}] Loading checkpoint from local: {local_path}")
        load_checkpoint(local_path, model, device=str(device))
    elif cfg.get("hf_repo_id"):
        repo   = cfg["hf_repo_id"]
        fname  = cfg.get("hf_filename", os.path.basename(local_path))
        dest   = os.path.dirname(local_path) if local_path else "outputs/checkpoints"
        print(f"[{label}] Local checkpoint not found. Pulling from HF: {repo}/{fname}")
        pulled = pull_from_hub(repo, fname, local_dir=dest)
        load_checkpoint(pulled, model, device=str(device))
    else:
        raise FileNotFoundError(
            f"[{label}] No checkpoint found at '{local_path}' and no hf_repo_id configured."
        )


# ---------------------------------------------------------------------------
# 1. load_models
# ---------------------------------------------------------------------------

def load_models(
    config: dict,
    device: torch.device,
) -> Tuple[SeqLabelLSTM, dict, object, object, object]:
    """
    Load Task1 LSTM and Task2 LM (SSM or BiLSTM).

    Returns
    -------
    lstm_model   : SeqLabelLSTM
    idx2plain    : dict  {int → char}
    plain_vocab  : Vocab (Task1)
    cipher_vocab : Vocab (Task1)
    lm_model     : SSMModel | BiLSTMModel
    lm_vocab     : VocabTask2
    """
    t1 = config["task1"]
    t2 = config["task2"]
    lm_type = t2["type"]          # "ssm" | "bilstm"

    # ---- System Alias for Pickle backwards compatibility ----
    # If the pickle file was created before the preprocessing folder was split
    # into task1/task2, pickle.load() will look for the old path.
    import src.preprocessing.task1.vocab
    sys.modules["src.preprocessing.vocab"] = src.preprocessing.task1.vocab

    # ---- Task 1 vocab + model ----------------------------------------
    cipher_vocab, plain_vocab = load_task1_vocab(t1["data_dir"])
    cipher_vocab_size = len(cipher_vocab)
    plain_vocab_size  = len(plain_vocab)
    pad_idx_plain     = plain_vocab.char2idx.get("<PAD>", 0)

    lstm_model = SeqLabelLSTM(
        config         = t1,
        cipher_vocab_size = cipher_vocab_size,
        plain_vocab_size  = plain_vocab_size,
        pad_idx           = pad_idx_plain,
    ).to(device)
    _load_ckpt(lstm_model, t1, "task1_lstm", device)
    lstm_model.eval()

    idx2plain = {v: k for k, v in plain_vocab.items()}

    # ---- Task 2 vocab + model ----------------------------------------
    _, _, lm_vocab = load_vocab_and_sequences(t2["data_dir"])
    vocab_size = lm_vocab.size

    if lm_type == "ssm":
        lm_model = SSMModel(
            vocab_size = vocab_size,
            embed_dim  = t2["embed_dim"],
            hidden_dim = t2["hidden_dim"],
            pad_idx    = lm_vocab.word2idx.get("<PAD>", 0),
            num_layers = t2.get("num_layers", 2),
            dropout    = t2.get("dropout", 0.1),
        ).to(device)
    else:   # bilstm
        lm_model = BiLSTMModel(
            vocab_size = vocab_size,
            embed_dim  = t2["embed_dim"],
            hidden_dim = t2["hidden_dim"],
            pad_idx    = lm_vocab.word2idx.get("<PAD>", 0),
            dropout    = t2.get("dropout", 0.2),
            num_layers = t2.get("num_layers", 1),
        ).to(device)

    _load_ckpt(lm_model, t2, f"task2_{lm_type}", device)
    lm_model.eval()

    return lstm_model, idx2plain, plain_vocab, cipher_vocab, lm_model, lm_vocab


# ---------------------------------------------------------------------------
# 2. preprocess_task1
# ---------------------------------------------------------------------------

def preprocess_task1(
    cipher_path: str,
    cipher_vocab,
    max_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Tokenise + encode a cipher test file using the frozen Task1 vocab.

    Returns
    -------
    tensors  : LongTensor [N, max_len]
    lengths  : LongTensor [N]
    raw_lines: list[str]
    """
    pad_idx = cipher_vocab.char2idx.get("<PAD>", 0)

    raw_lines: List[str] = []
    with open(cipher_path, "r", encoding="utf-8") as f:
        for line in f:
            raw_lines.append(line.rstrip("\n"))

    if not raw_lines:
        return (
            torch.empty((0, max_len), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            raw_lines,
        )

    tensors: List[torch.Tensor] = []
    lengths: List[int] = []

    for line in tqdm(raw_lines, desc="Prep Task1 Tokens", leave=False):
        tokens = tokenize_cipher_line(line)
        if tokens is None:
            tokens = []
        if not tokens:
            tokens = ["<PAD>"]

        indices = cipher_vocab.encode(tokens)
        length  = min(len(indices), max_len)
        indices = indices[:max_len]
        indices += [pad_idx] * (max_len - len(indices))

        tensors.append(torch.tensor(indices, dtype=torch.long))
        lengths.append(length)

    return (
        torch.stack(tensors).to(device),
        torch.tensor(lengths, dtype=torch.long).to(device),
        raw_lines,
    )


# ---------------------------------------------------------------------------
# 3. run_lstm
# ---------------------------------------------------------------------------

def run_lstm(
    model: SeqLabelLSTM,
    tensors: torch.Tensor,
    lengths: torch.Tensor,
    idx2plain: dict,
    pad_idx: int,
    batch_size: int = 64,
) -> List[str]:
    """
    Batched LSTM inference → list of decoded plain-text strings.
    """
    results: List[str] = []
    model.eval()

    with torch.no_grad():
        for start in tqdm(range(0, len(tensors), batch_size), desc="LSTM Decryption", leave=False):
            batch_c = tensors[start : start + batch_size]
            batch_l = lengths[start : start + batch_size]

            logits = model(batch_c, batch_l)           # [B, T, plain_vocab]
            preds  = logits.argmax(dim=-1)             # [B, T]

            for i in range(preds.shape[0]):
                act_len = batch_l[i].item()
                text    = indices_to_string(preds[i][:act_len], idx2plain, pad_idx)
                results.append(text)

    return results


# ---------------------------------------------------------------------------
# 4. preprocess_task2
# ---------------------------------------------------------------------------

def preprocess_task2(
    plain_lines: List[str],
    vocab,
    max_words: int = 20,
) -> Tuple[List[List[str]], torch.Tensor]:
    """
    Convert decoded plain-text lines to Task2 vocab token lists AND
    chunked+padded tensors (for perplexity computation only).

    Returns
    -------
    word_lists : list of token lists (one per line, for correction loop)
    chunks     : LongTensor [TotalChunks, max_words]   (for perplexity)
    """
    pad_idx = vocab.word2idx.get("<PAD>", 0)
    eos_idx = vocab.word2idx.get("<EOS>", 3)

    word_lists: List[List[str]] = []
    all_chunks: List[List[int]] = []

    for line in tqdm(plain_lines, desc="Prep Task2 Tokens", leave=False):
        tokens = tokenize_line(line)      # lowercase + split
        word_lists.append(tokens)

        encoded = vocab.encode(tokens) + [eos_idx]
        for i in range(0, len(encoded), max_words):
            chunk = encoded[i : i + max_words]
            if len(chunk) < max_words:
                chunk = chunk + [pad_idx] * (max_words - len(chunk))
            all_chunks.append(chunk)

    chunks_tensor = torch.tensor(all_chunks, dtype=torch.long) if all_chunks else torch.zeros((0, max_words), dtype=torch.long)
    return word_lists, chunks_tensor


# ---------------------------------------------------------------------------
# 5. correct_with_ssm
# ---------------------------------------------------------------------------

def correct_with_ssm(
    model: SSMModel,
    words: List[str],
    vocab,
    device: torch.device,
    threshold: float = 0.05,
) -> List[str]:
    """
    Token-level SSM correction (causal Next Word Prediction).

    For position i >= 1:
        context = corrected[0:i]
        logits  = model(encoded_context)          shape [1, i, V]
        probs   = softmax(logits[0, i-1, :])      P(next | context)
        if probs[word_i] < threshold → replace with argmax

    Position 0 is skipped: no previous context exists.
    PAD and EOS tokens are never replaced.
    """
    corrected = list(words)
    pad_idx   = vocab.word2idx.get("<PAD>", 0)
    eos_idx   = vocab.word2idx.get("<EOS>", 3)
    skip_ids  = {pad_idx, eos_idx}

    model.eval()
    with torch.no_grad():
        for i in range(1, len(corrected)):          # skip i=0 (no context)
            word = corrected[i]
            word_id = vocab.encode([word])[0]

            if word_id in skip_ids:                 # never correct PAD / EOS
                continue

            # Build context [0:i]
            context_ids = vocab.encode(corrected[:i])
            if not context_ids:
                continue

            inp    = torch.tensor([context_ids], dtype=torch.long, device=device)  # [1, i]
            logits = model(inp)                                                      # [1, i, V]
            probs  = F.softmax(logits[0, -1, :], dim=-1)                            # [V] — P(next)

            confidence = probs[word_id].item()
            best_id    = probs.argmax().item()
            best_prob  = probs[best_id].item()

            unk_idx = vocab.word2idx.get("<UNK>", 1)

            if word_id == unk_idx or (confidence < threshold and best_prob > confidence):
                if best_id != unk_idx:
                    corrected[i]  = vocab.idx2word.get(best_id, word)

    return corrected


# ---------------------------------------------------------------------------
# 6. correct_with_bilstm
# ---------------------------------------------------------------------------

def correct_with_bilstm(
    model: BiLSTMModel,
    words: List[str],
    vocab,
    device: torch.device,
    threshold: float = 0.05,
) -> List[str]:
    """
    Optimized Token-level BiLSTM correction (Masked Language Model).
    Uses a 2-pass confidence strategy to eliminate O(N^2) evaluation:
    1. One forward pass unmasked to identify low-confidence / UNK tokens.
    2. Sequentially mask and re-evaluate only those flagged positions.
    """
    corrected  = list(words)
    pad_idx    = vocab.word2idx.get("<PAD>", 0)
    mask_idx   = vocab.word2idx.get("<MASK>", 2)
    eos_idx    = vocab.word2idx.get("<EOS>", 3)
    unk_idx    = vocab.word2idx.get("<UNK>", 1)
    skip_ids   = {pad_idx, mask_idx, eos_idx}

    model.eval()
    with torch.no_grad():
        # Pass 1: Encode full unmasked sequence
        base_ids = vocab.encode(corrected)
        if not base_ids:
            return corrected
            
        inp = torch.tensor([base_ids], dtype=torch.long, device=device)
        base_logits = model(inp)  # [1, len, V]
        base_probs = F.softmax(base_logits[0], dim=-1)  # [len, V]
        
        # Identify weak positions
        positions_to_eval = []
        for i in range(len(corrected)):
            word_id = base_ids[i]
            if word_id in skip_ids:
                continue
                
            confidence = base_probs[i, word_id].item()
            if word_id == unk_idx or confidence < threshold:
                positions_to_eval.append(i)
                
        # Pass 2: Exact masking ONLY for flagged positions (processed simultaneously)
        if positions_to_eval:
            masked_ids = list(base_ids)
            for i in positions_to_eval:
                masked_ids[i] = mask_idx
            
            m_inp = torch.tensor([masked_ids], dtype=torch.long, device=device)
            m_logits = model(m_inp)
            m_probs = F.softmax(m_logits[0], dim=-1)  # [len, V]
            
            for i in positions_to_eval:
                word_id = base_ids[i]
                word_str = corrected[i]
                
                m_conf = m_probs[i, word_id].item()
                best_id = m_probs[i].argmax().item()
                best_prob = m_probs[i, best_id].item()
                
                if word_id == unk_idx or (m_conf < threshold and best_prob > m_conf):
                    if best_id != unk_idx:
                        fixed_word = vocab.idx2word.get(best_id, word_str)
                        corrected[i] = fixed_word
                        base_ids[i] = best_id  # Update reference for structural integrity

    return corrected


# ---------------------------------------------------------------------------
# 7. compute_metrics (wrapper)
# ---------------------------------------------------------------------------

def compute_metrics(predictions: List[str], targets: List[str], perplexity: float = 0.0) -> dict:
    """Run all Task3 metrics against ground truth."""
    metrics = compute_all_metrics(predictions, targets)
    metrics["perplexity"] = perplexity
    return metrics


# ---------------------------------------------------------------------------
# 8. run_experiment
# ---------------------------------------------------------------------------

def run_experiment(config: dict, lm_type: str) -> None:
    """
    Main orchestration loop:
        for each cipher file → LSTM-only run + LSTM+LM run → metrics + files.
    """
    device = _get_device()
    t1_cfg = config["task1"]
    t2_cfg = config["task2"]
    out_dir = config.get("output_dir", "outputs/results")
    threshold = config.get("threshold", 0.05)
    max_words = config.get("max_words", 20)
    batch_size = t1_cfg.get("batch_size", 64)

    overall_start_time = time.time()

    # Infer max_len from the Task1 preprocessed dataset tensor
    task1_data_path = os.path.join(t1_cfg["data_dir"], "dataset.pt")
    task1_data      = torch.load(task1_data_path, weights_only=True)
    max_len         = int(task1_data.get("max_len", task1_data["input"].shape[1]))

    # ---- Load ground truth --------------------------------------------------
    gt_path = config.get("ground_truth", "data/plain.txt")
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth_lines = [line.rstrip("\n") for line in f]
    print(f"[Task3] Ground truth loaded: {len(ground_truth_lines)} lines from {gt_path}")

    # ---- Load models (once, shared across all cipher files) -----------------
    print(f"\n[Task3] Loading models (lm_type={lm_type}) ...")
    lstm_model, idx2plain, plain_vocab, cipher_vocab, lm_model, lm_vocab = load_models(
        config, device
    )
    pad_idx_plain = plain_vocab.char2idx.get("<PAD>", 0)
    pad_idx_lm    = lm_vocab.word2idx.get("<PAD>", 0)
    mask_idx_lm   = lm_vocab.word2idx.get("<MASK>", 2)

    summary_rows: List[dict] = []
    os.makedirs(out_dir, exist_ok=True)

    # ---- Loop over cipher files ---------------------------------------------
    for cipher_path in config.get("cipher_files", []):
        file_start_time = time.time()
        
        cipher_tag = os.path.splitext(os.path.basename(cipher_path))[0]  # e.g. "cipher_01"
        print(f"\n{'='*60}")
        print(f"[Task3] Processing: {cipher_path}  (tag={cipher_tag})")
        print(f"{'='*60}")

        # Step 1 — Task1 preprocessing
        tensors, lengths, raw_lines = preprocess_task1(
            cipher_path, cipher_vocab, max_len, device
        )
        n_lines = len(raw_lines)
        print(f"[Task3]   {n_lines} cipher lines encoded.")

        # Select matching ground truth slice (lines are parallel by design)
        gt_slice = ground_truth_lines[:n_lines]

        # Step 2 — Run LSTM decryption
        print(f"[Task3]   Running LSTM decryption ...")
        lstm_outputs: List[str] = run_lstm(
            lstm_model, tensors, lengths, idx2plain, pad_idx_plain, batch_size
        )

        # Step 3 & 4 — Transition to Task2 space and compute LSTM-only metrics
        print(f"[Task3]   Converting to Task2 token space ...")
        word_lists, chunks_tensor = preprocess_task2(lstm_outputs, lm_vocab, max_words)

        ppl_lstm = float("inf")
        if chunks_tensor.shape[0] > 0:
            if chunks_tensor.shape[0] > 10000:
                torch.manual_seed(42)  # Deterministic evaluation
                idx = torch.randperm(chunks_tensor.shape[0])[:10000]
                eval_chunks = chunks_tensor[idx]
            else:
                eval_chunks = chunks_tensor
                
            if lm_type == "ssm":
                ppl_lstm = ssm_perplexity(lm_model, eval_chunks, pad_idx_lm, device)
            else:
                ppl_lstm = bilstm_perplexity(lm_model, eval_chunks, mask_idx_lm, pad_idx_lm, device)

        metrics_lstm = compute_metrics(lstm_outputs, gt_slice, perplexity=ppl_lstm)
        print(f"[Task3]   [LSTM only] char_acc={metrics_lstm['char_accuracy']:.4f}  "
              f"word_acc={metrics_lstm['word_accuracy']:.4f}  "
              f"bleu={metrics_lstm['bleu']:.4f}  "
              f"ppl={metrics_lstm['perplexity']:.2f}")

        # Step 5 — LM correction (per sentence)
        print(f"[Task3]   Running LM correction ({lm_type}, threshold={threshold}) ...")
        corrected_lines: List[str] = []
        for words in tqdm(word_lists, desc=f"LM Correction ({lm_type.upper()})", leave=False):
            if not words:
                corrected_lines.append("")
                continue
            if lm_type == "ssm":
                fixed = correct_with_ssm(lm_model, words, lm_vocab, device, threshold)
            else:
                fixed = correct_with_bilstm(lm_model, words, lm_vocab, device, threshold)
            corrected_lines.append(" ".join(fixed))

        # Step 6 — LSTM + LM metrics
        _, chunks_corrected = preprocess_task2(corrected_lines, lm_vocab, max_words)
        ppl_corrected = float("inf")
        if chunks_corrected.shape[0] > 0:
            if chunks_corrected.shape[0] > 10000:
                torch.manual_seed(42)  # Deterministic evaluation
                idx = torch.randperm(chunks_corrected.shape[0])[:10000]
                eval_chunks_corr = chunks_corrected[idx]
            else:
                eval_chunks_corr = chunks_corrected
                
            if lm_type == "ssm":
                ppl_corrected = ssm_perplexity(lm_model, eval_chunks_corr, pad_idx_lm, device)
            else:
                ppl_corrected = bilstm_perplexity(lm_model, eval_chunks_corr, mask_idx_lm, pad_idx_lm, device)

        metrics_corrected = compute_metrics(corrected_lines, gt_slice, perplexity=ppl_corrected)
        print(f"[Task3]   [LSTM+{lm_type.upper()}] char_acc={metrics_corrected['char_accuracy']:.4f}  "
              f"word_acc={metrics_corrected['word_accuracy']:.4f}  "
              f"bleu={metrics_corrected['bleu']:.4f}  "
              f"ppl={metrics_corrected['perplexity']:.2f}")

        # Step 7 — Write per-file output files
        lstm_only_path      = os.path.join(out_dir, f"task3_{cipher_tag}_lstm_only.txt")
        lm_corrected_path   = os.path.join(out_dir, f"task3_{cipher_tag}_lstm_{lm_type}.txt")

        _write_output_file(lstm_only_path, lstm_outputs, gt_slice, metrics_lstm, "LSTM Only")
        _write_output_file(lm_corrected_path, corrected_lines, gt_slice, metrics_corrected,
                           f"LSTM + {lm_type.upper()}")

        print(f"[Task3]   Output written:")
        print(f"[Task3]     {lstm_only_path}")
        print(f"[Task3]     {lm_corrected_path}")

        file_end_time = time.time()
        print(f"[Task3]   FINISHED {cipher_tag} in {file_end_time - file_start_time:.1f} seconds.\n")

        summary_rows.append({
            "cipher_file": cipher_path,
            "lstm_only":   metrics_lstm,
            f"lstm_{lm_type}": metrics_corrected,
        })

    # ---- Write combined summary file ----------------------------------------
    _write_summary(config, summary_rows, lm_type, out_dir)
    
    overall_end_time = time.time()
    print(f"\n[Task3] ===========================================================")
    print(f"[Task3] TOTAL TASK 3 TIME ({lm_type.upper()}): {overall_end_time - overall_start_time:.1f} seconds")
    print(f"[Task3] ===========================================================")


# ---------------------------------------------------------------------------
# Output file writers
# ---------------------------------------------------------------------------

def _write_output_file(
    path: str,
    predictions: List[str],
    targets: List[str],
    metrics: dict,
    label: str,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        # Write the full pipeline output line-by-line
        for pred in predictions:
            f.write(f"{pred}\n")
            
        # Append all metrics at the bottom
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"METRICS: {label}\n")
        f.write("=" * 60 + "\n")
        for k, v in metrics.items():
            f.write(f"  {k:25s}: {v:.6f}\n")


def _write_summary(config: dict, rows: List[dict], lm_type: str, out_dir: str) -> None:
    summary_path = os.path.join(out_dir, "task3_output.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"  TASK 3 EXPERIMENT SUMMARY  (lm_type={lm_type})\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        metric_keys = ["char_accuracy", "word_accuracy", "bleu", "rouge1", "rougeL", "avg_levenshtein", "perplexity"]

        for row in rows:
            cfile = row["cipher_file"]
            f.write(f"--- {cfile} ---\n")
            f.write(f"{'Metric':<25} {'LSTM only':>12}  {'LSTM+' + lm_type.upper():>14}\n")
            f.write("-" * 55 + "\n")
            lstm_m = row["lstm_only"]
            lm_m   = row.get(f"lstm_{lm_type}", {})
            for k in metric_keys:
                f.write(f"  {k:<23} {lstm_m.get(k, 0):>12.4f}  {lm_m.get(k, 0):>14.4f}\n")
            f.write("\n")

    print(f"\n[Task3] Combined summary written: {summary_path}")
    # Also save JSON for programmatic access
    json_path = summary_path.replace(".txt", ".json")
    with open(json_path, "w") as jf:
        json.dump(rows, jf, indent=2)
    print(f"[Task3] JSON summary written:     {json_path}")


# ---------------------------------------------------------------------------
# Public entry point (called by main.py)
# ---------------------------------------------------------------------------

def main(config_path: str, mode: str) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    lm_type = config["task2"]["type"]   # "ssm" or "bilstm"
    print(f"[Task3] Config: {config_path}  lm_type={lm_type}  mode={mode}")

    # Task 3 has no training — both "train" and "evaluate" run the experiment
    run_experiment(config, lm_type)
