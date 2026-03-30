import os
import random
import yaml
import torch
from src.preprocessing.task2.tokenizer import tokenize_line
from src.preprocessing.task2.vocab import VocabTask2
from src.preprocessing.task2.padder import pad_sequence
from src.preprocessing.task2.save_load import save_cache

def main():
    config_path = "config/task2/dataset.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_path = config.get("data_path", "data/plain.txt")
    save_dir = config.get("save_dir", "data/processed/task2/")
    max_words = config.get("max_words", 20)
    min_freq = config.get("min_freq", 1)
    force = config.get("force_reprocess", False)
    val_split = config.get("val_split", 0.1)
    seed = config.get("seed", 42)

    dataset_file = os.path.join(save_dir, "dataset_task2.pt")
    vocab_file = os.path.join(save_dir, "vocab_task2.pkl")
    if os.path.exists(dataset_file) and os.path.exists(vocab_file) and not force:
        print("[Task2] Cache exists. Use force_reprocess: true to rebuild.")
        return

    # 1. Load data
    # 2. Lowercase & 3. Tokenize
    print("[Task2] Loading and tokenizing sentences...")
    tokenized_sentences = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = tokenize_line(line)
            if tokens:
                tokenized_sentences.append(tokens)

    # NEW: Shuffle and split at the sentence level
    random.seed(seed)
    random.shuffle(tokenized_sentences)
    
    n_total = len(tokenized_sentences)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val
    
    train_sentences = tokenized_sentences[:n_train]
    val_sentences   = tokenized_sentences[n_train:]

    # 4. Build vocabulary (ONLY on train_sentences)
    print(f"[Task2] Building vocabulary on {n_train} training sentences...")
    vocab = VocabTask2()
    # Included common special tokens
    special_tokens = ["<PAD>", "<UNK>", "<MASK>", "<EOS>"]
    vocab.build_vocab(train_sentences, special_tokens=special_tokens, min_freq=min_freq)

    # 5. Encode tokens -> indices
    # 6. Apply sentence-based chunking
    # 7. Pad sequences to length 20
    print(f"[Task2] Encoding, chunking (max {max_words}), and padding...")
    pad_idx = vocab.word2idx["<PAD>"]
    eos_idx = vocab.word2idx["<EOS>"]
    
    def process_sentences(sentences):
        chunks = []
        for tokens in sentences:
            encoded = vocab.encode(tokens) + [eos_idx]
            # Split into multiple chunks of size max_words if longer
            for i in range(0, len(encoded), max_words):
                chunk = encoded[i : i + max_words]
                chunk_padded = pad_sequence(chunk, max_words, pad_idx)
                chunks.append(chunk_padded)
        return torch.tensor(chunks, dtype=torch.long)
        
    train_tensor = process_sentences(train_sentences)
    val_tensor   = process_sentences(val_sentences)

    # 8. Cache results
    print(f"[Task2] Saving sequence chunks (Train: {len(train_tensor)}, Val: {len(val_tensor)}) to {save_dir}...")
    save_cache({
        "train_sequences": train_tensor,
        "val_sequences": val_tensor
    }, vocab, save_dir)
    print(f"[Task2] Preprocessing complete. Vocab size: {vocab.size}")

if __name__ == "__main__":
    main()
