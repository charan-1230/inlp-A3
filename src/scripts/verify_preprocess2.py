import yaml
import torch
from torch.utils.data import DataLoader

from src.task2.utils import load_vocab_and_sequences
from src.task2.dataset_mlm import MLMDataset

def decode_seq(seq, vocab):
    return [vocab.idx2word.get(int(i), "<UNK>") for i in seq]

def check_padding(seq, pad_idx):
    if pad_idx in seq:
        first_pad = (seq == pad_idx).nonzero(as_tuple=True)[0][0].item()
        assert all(x == pad_idx for x in seq[first_pad:]), "Padding not contiguous at end"

def main():
    config_path = "config/task2/dataset.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("\n[TEST] Loading preprocessed train/val datasets...")

    train_sequences, val_sequences, vocab = load_vocab_and_sequences()

    print(f"[TEST] Train shape: {train_sequences.shape}")
    print(f"[TEST] Val shape:   {val_sequences.shape}")
    print(f"[TEST] Vocab size: {vocab.size}")

    # -------------------------
    # Basic shape checks
    # -------------------------
    assert len(train_sequences.shape) == 2
    assert len(val_sequences.shape) == 2

    print("[TEST] Shape checks passed")

    pad_idx = vocab.word2idx["<PAD>"]
    unk_idx = vocab.word2idx["<UNK>"]

    # -------------------------
    # Padding check (train + val)
    # -------------------------
    print("\n[TEST] Checking padding...")

    check_padding(train_sequences[0], pad_idx)
    check_padding(val_sequences[0], pad_idx)

    print("[TEST] Padding check passed")

    # -------------------------
    # Sample decoding
    # -------------------------
    print("\n[TEST] Sample train decoded:")
    print(decode_seq(train_sequences[0], vocab))

    print("\n[TEST] Sample val decoded:")
    print(decode_seq(val_sequences[0], vocab))

    # -------------------------
    # UNK check (important after fix)
    # -------------------------
    print("\n[TEST] Checking UNK presence in validation set...")

    unk_count = (val_sequences == unk_idx).sum().item()

    print(f"[TEST] UNK tokens in validation: {unk_count}")

    assert unk_count > 0, "No UNK tokens found — vocab leakage likely still exists"

    print("[TEST] UNK check passed (no vocab leakage)")

    # -------------------------
    # MLM Dataset test (train)
    # -------------------------
    print("\n[TEST] Testing MLM dataset (train)...")

    mlm_dataset = MLMDataset({"sequences": train_sequences}, vocab, config)

    loader = DataLoader(mlm_dataset, batch_size=2, shuffle=False)

    inputs, labels = next(iter(loader))

    print("[TEST] MLM batch input shape:", inputs.shape)
    print("[TEST] MLM batch label shape:", labels.shape)

    print("\n[TEST] Sample MLM input:")
    print(inputs[0].tolist())

    print("\n[TEST] Sample MLM labels:")
    print(labels[0].tolist())

    # -------------------------
    # Mask correctness check
    # -------------------------
    masked_positions = (labels[0] != -100).nonzero(as_tuple=True)[0]

    print("\n[TEST] Masked positions:", masked_positions.tolist())

    for pos in masked_positions:
        original = train_sequences[0][pos].item()
        label    = labels[0][pos].item()

        assert label == original, "Masked label mismatch"

    print("[TEST] MLM masking check passed")

    print("\n✅ ALL TESTS PASSED SUCCESSFULLY")

if __name__ == "__main__":
    main()
