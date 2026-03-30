# Assignment 3: Report

---

## 1. Data Preprocessing

Preprocessing lives under `src/preprocessing/` (split into `task1/` and `task2/`). Both pipelines produce cached `.pt` tensors and `.pkl` vocabularies under `data/processed/` so re-tokenization is never needed at training time.

### 1.1 Task 1 Preprocessing — Cipher to Plaintext

**Cipher format:** `9` encodes a space (`<SPACE>` token); every pair of consecutive non-9 digits (e.g. `31`, `47`) encodes one plaintext character. A cipher line `3147 9...` becomes `['31', '47', '<SPACE>', ...]`.

**Tokenization (`tokenizer.py`):** `tokenize_cipher_line` scans character by character, emitting 2-digit tokens or `<SPACE>`. Lines with a dangling single digit or a `9` inside a 2-digit window are rejected. `tokenize_plain_line` returns `list(line)` — individual characters including spaces.

**Alignment filter:** A pair is kept only if (1) tokenization succeeds, (2) `len(cipher_tokens) == len(plain_tokens)`, and (3) every `<SPACE>` cipher token aligns to a literal space in the plaintext.

**Vocabularies (`vocab.py`):**
- *Cipher vocab*: special tokens `<PAD>`, `<UNK>`, `<SPACE>` first, then all unique 2-digit tokens sorted.
- *Plain vocab*: `<PAD>`, `<UNK>` first, then all observed characters sorted.
- No `<SOS>`/`<EOS>` — sequence labeling labels every position independently.

**Encoding & padding (`dataset_builder.py`, `padder.py`):** Sequences are encoded to integer indices and truncated at `MAX_LEN=200`. All sequences are then right-padded with `<PAD>` to the same length; original lengths are stored in a `lengths` tensor used to mask padding during the forward pass.

**Saved artifacts:** `dataset.pt` (`input`, `target`, `lengths`, `max_len`), `cipher_vocab.pkl`, `plain_vocab.pkl` → `data/processed/task1/`.

### 1.2 Task 2 Preprocessing — Plaintext for Language Modeling

**Tokenization (`tokenizer.py`):** Each line of `plain.txt` is lowercased and split on spaces; empty tokens are dropped.

**Split before vocab build:** Sentences are shuffled (`seed=42`), then split 90/10. Vocabulary is built **only on training sentences** to avoid data leakage.

**Vocabulary (`vocab.py`):** Fixed special token order — `<PAD>`(0), `<UNK>`(1), `<MASK>`(2), `<EOS>`(3). Words with `freq < 2` (`min_freq=2`) are excluded; rare words map to `<UNK>` at encode time.

**Encoding & chunking:** Each sentence is encoded, an `<EOS>` is appended, then sliced into fixed windows of `max_words=20`. Short final chunks are padded to 20 with `<PAD>`.

**Dataset-specific usage:**
- *SSM (`dataset_nwp.py`)*: `input = seq[:-1]`, `target = seq[1:]` — standard next-word prediction shift.
- *BiLSTM (`dataset_mlm.py`)*: Dynamic BERT-style masking at `__getitem__` time (not cached). 15% of non-`<PAD>` tokens selected; of those, 80% replaced with `<MASK>`, 10% with a random valid token, 10% left unchanged. Labels are original token IDs at masked positions, `-100` elsewhere.

**Saved artifacts:** `dataset_task2.pt` (`train_sequences`, `val_sequences`), `vocab_task2.pkl` → `data/processed/task2/`. MLM masks are NOT cached; only base token sequences are stored.

### 1.3 Task 3 Preprocessing — Runtime Tokenization

No offline preprocessing step. Task 3 reuses frozen Task 1 and Task 2 vocabularies from disk:
- **Cipher → Task 1 space**: `tokenize_cipher_line()` + cipher vocab + pad to `max_len` from `dataset.pt`. Failed lines fall back to `<PAD>`.
- **LSTM output → Task 2 space**: `tokenize_line()` (lowercase/split) + Task 2 vocab encode. Unknown words become `<UNK>`, which the correction logic prioritizes.

---

## 2. Methodology

### 2.1 Task 1 — Cipher Decryption

Both models treat decryption as **character-level sequence labeling**: each cipher token is independently mapped to a plaintext character via a softmax over the plain vocabulary.

**RNN:** Each cell computes `h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)`. The single nonlinearity squashes all history, causing vanishing gradients on long sequences and limiting effective context to nearby positions.

**LSTM:** Replaces the single gate with four: input (i), forget (f), cell (g), output (o). The forget gate selectively erases cell state, keeping gradients alive across long character spans. Implemented from scratch (no `nn.LSTM`), stacked at `num_layers=2` with inter-layer dropout. Final hidden states are projected to per-character logits over the plain vocabulary.

The LSTM's gated cell state is the key advantage: it can carry cipher-mapping context across long sequences and reset when the structure shifts, which the RNN cannot do reliably.

### 2.2 Task 2 — Language Modeling

**SSM (causal NWP):** A stacked LSTM predicting word t+1 from words 1..t. Only left context is available; perplexity = exp(mean CE loss on shifted targets). In Task 3, the SSM acts as an autoregressive scorer over the corrected prefix.

**BiLSTM (MLM):** Trained with BERT-style masked prediction. The hidden state at position t concatenates a forward and backward LSTM pass, giving access to full sentence context. This makes it more expressive than the SSM for correction tasks, especially at mid-sentence positions where right context resolves ambiguity.

### 2.3 Task 3 — Error Correction Pipeline

No retraining. For each cipher file, the pipeline:
1. Decrypts cipher → LSTM → noisy plaintext.
2. Tokenizes decoded text into Task 2 word space.
3. Applies LM-based token-level correction.
4. Evaluates LSTM-only and LSTM+LM outputs vs. ground truth.

**Confidence-based correction:** A word at position i is replaced only if its LM-assigned probability < `threshold=0.05` AND the model's top prediction has higher probability, OR the word is `<UNK>` (always replaced). `<PAD>` and `<EOS>` are never touched.

**SSM correction:** Sequential left-to-right. At position i, the corrected prefix [0:i] is fed as context; the next-word distribution scores position i. Position 0 is skipped (no context). Risk: early errors propagate to later corrections.

**BiLSTM correction (2-pass):** Pass 1 — full unmasked forward pass identifies low-confidence/`<UNK>` positions. Pass 2 — all flagged positions are masked simultaneously in one forward pass; predictions at masked positions are used for replacement. This avoids O(N²) inference and exploits full bidirectional context per correction.

---

## 3. Experimental Setup

### 3.1 Dataset

`data/plain.txt` is the single plaintext source. Task 1 uses `cipher_00.txt` for training. Task 3 runs on `cipher_01.txt`–`cipher_04.txt` with `plain.txt` as ground truth.

### 3.2 Train / Validation Split

90/10 split, `seed=42`, deterministic across all tasks.

### 3.3 Perplexity Computation

- **SSM**: Standard NWP — exp(mean CE loss over non-PAD positions on shifted 20-word chunks).
- **BiLSTM**: Batched 15% masked evaluation with `seed=42` for reproducibility. At least one token per sequence is forced masked. If chunks exceed 10,000, a random sample of 10,000 is used.


## 4. Final Configurations

### Task 1

| Hyperparameter  | RNN    | LSTM   |
|-----------------|:------:|:------:|
| `embed_dim`     | 128    | 128    |
| `hidden_dim`    | 512    | 512    |
| `num_layers`    | 2      | 2      |
| `dropout`       | 0.2    | 0.3    |
| `learning_rate` | 0.002  | 0.001  |
| `batch_size`    | 256    | 256    |
| `max_epochs`    | 80     | 80     |
| `patience`      | 8      | 8      |
| `grad_clip`     | 1.0    | 1.0    |

LSTM uses lower LR (0.001) for stable four-gate optimization and higher dropout (0.3) to prevent over-fitting given its larger parameter count.

### Task 2

| Hyperparameter  | SSM    | BiLSTM |
|-----------------|:------:|:------:|
| `embed_dim`     | 256    | 192    |
| `hidden_dim`    | 320    | 384    |
| `num_layers`    | 2      | 2      |
| `dropout`       | 0.35   | 0.3    |
| `learning_rate` | 0.0004 | 0.0005 |
| `batch_size`    | 32     | 64     |
| `max_epochs`    | 30     | 20     |
| `patience`      | 6      | 3      |
| `mask_prob`     | N/A    | 0.15   |

BiLSTM uses a tighter patience (3) because MLM plateaus faster once bidirectional context is learned.

---

## 5. Results

### 5.1 Task 1 Results

| Model | Char Accuracy | Word Accuracy | Avg Levenshtein |
|-------|:-------------:|:-------------:|:---------------:|
| RNN   | 0.7762        | 0.3512        | 21.10           |
| LSTM  | **0.9099**    | **0.4980**    | **8.51**        |

LSTM improves char accuracy by ~14 points, nearly doubles word accuracy, and cuts Levenshtein distance by more than half. Short sentences (e.g., "Their father is Charles B Armour") decode perfectly; errors concentrate in long sequences where the hidden state drifts — consistent with the character-level independence assumption of sequence labeling. The RNN's high Levenshtein (21.1) indicates structural divergence, not just isolated errors.
---

### 5.2 Task 2 Results

| Model  | Val Loss | Val Perplexity |
|--------|:--------:|:--------------:|
| SSM    | 5.6473   | 283.51         |
| BiLSTM | **4.9672** | **143.63**   |

BiLSTM achieves ~2× lower perplexity. The SSM's higher perplexity reflects the harder causal inference problem and lack of right context. Sample outputs confirm this: SSM collapses to high-frequency words ("the", "a"), while BiLSTM produces more contextually coherent substitutions ("possible" for "necessary").

---

### 5.3 Task 3 Results

Only LSTM+BiLSTM was evaluated (SSM pipeline results not available).

#### cipher_01.txt

| Metric          | LSTM Only | LSTM + BiLSTM |
|-----------------|:---------:|:-------------:|
| Char Accuracy   | 0.7716    | 0.1199        |
| Word Accuracy   | 0.0142    | 0.0003        |
| Avg Levenshtein | 22.14     | 52.23         |
| BLEU            | 0.1880    | 0.1004        |
| ROUGE-1         | 0.5398    | 0.4506        |
| ROUGE-L         | 0.5394    | 0.4425        |
| Perplexity      | 54.58     | 60.50         |

#### cipher_02.txt

| Metric          | LSTM Only | LSTM + BiLSTM |
|-----------------|:---------:|:-------------:|
| Char Accuracy   | 0.6949    | 0.1080        |
| Word Accuracy   | 0.0140    | 0.0003        |
| Avg Levenshtein | 29.74     | 57.00         |
| BLEU            | 0.0707    | 0.0544        |
| ROUGE-1         | 0.3999    | 0.3738        |
| ROUGE-L         | 0.3993    | 0.3624        |
| Perplexity      | 35.04     | 96.58         |

#### cipher_03.txt

| Metric          | LSTM Only | LSTM + BiLSTM |
|-----------------|:---------:|:-------------:|
| Char Accuracy   | 0.6292    | 0.1021        |
| Word Accuracy   | 0.0140    | 0.0003        |
| Avg Levenshtein | 36.24     | 60.32         |
| BLEU            | 0.0308    | 0.0322        |
| ROUGE-1         | 0.3162    | 0.3234        |
| ROUGE-L         | 0.3155    | 0.3103        |
| Perplexity      | 24.47     | 164.89        |

#### cipher_04.txt

| Metric          | LSTM Only | LSTM + BiLSTM |
|-----------------|:---------:|:-------------:|
| Char Accuracy   | 0.5745    | 0.0986        |
| Word Accuracy   | 0.0139    | 0.0003        |
| Avg Levenshtein | 41.62     | 62.60         |
| BLEU            | 0.0147    | 0.0205        |
| ROUGE-1         | 0.2636    | 0.2891        |
| ROUGE-L         | 0.2627    | 0.2754        |
| Perplexity      | 19.07     | 257.38        |

#### Noise-Level Summary

| Cipher File   | LSTM Char Acc | BiLSTM Char Acc | LSTM Perplexity | BiLSTM Perplexity |
|---------------|:-------------:|:---------------:|:---------------:|:-----------------:|
| cipher_01.txt | 0.7716        | 0.1199          | 54.58           | 60.50             |
| cipher_02.txt | 0.6949        | 0.1080          | 35.04           | 96.58             |
| cipher_03.txt | 0.6292        | 0.1021          | 24.47           | 164.89            |
| cipher_04.txt | 0.5745        | 0.0986          | 19.07           | 257.38            |

**Analysis:** The BiLSTM correction causes a severe drop in character accuracy (~0.57–0.77 → ~0.10) and near-zero word accuracy across all cipher files. This is primarily a **vocabulary mismatch** problem: the LSTM operates at the character level and its noisy decoded output (especially at higher noise levels) produces many words absent from the Task 2 vocabulary. These decode to `<UNK>`, which the pipeline always replaces. At high error rates, a large fraction of each decoded sentence is `<UNK>`, so the BiLSTM effectively rewrites entire sentences with fluent but incorrect words. ROUGE-1 and BLEU partially hold up because some word-overlap is preserved, but character-level and word-accuracy metrics collapse. The escalating BiLSTM perplexity across ciphers (60 → 257) confirms this cascading rewrite worsens as LSTM error rate increases. This highlights a fundamental limitation: LM-based correction only works when the majority of decoded tokens are already correct in-vocabulary words.

---

## 9.Links

- **HuggingFace**: (https://huggingface.co/Charan0530/inlp-a3)
- **WandB — Task 1**: https://wandb.ai/saicharanbakaram30-iiit-hyderabad/inlp-a3-task1
- **WandB — Task 2**: https://wandb.ai/saicharanbakaram30-iiit-hyderabad/inlp-a3-task2