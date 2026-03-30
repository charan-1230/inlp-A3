import os
import torch

from src.preprocessing.task1.tokenizer import load_data, tokenize_cipher_line, tokenize_plain_line, SPACE_TOKEN
from src.preprocessing.task1.vocab import Vocab
from src.preprocessing.task1.dataset_builder import build_dataset, pad_dataset
from src.preprocessing.task1.save_load import save_preprocessed


def main():
    plain_path = "data/plain.txt"
    cipher_path = "data/cipher_00.txt"
    save_dir = "data/processed/task1/"

    # 1. Load data
    plain_lines, cipher_lines = load_data(plain_path, cipher_path)

    # 2. Tokenize and filter aligned pairs
    aligned_cipher_tokens = []
    aligned_plain_tokens = []
    num_skipped = 0

    for c_line, p_line in zip(cipher_lines, plain_lines):
        c_toks = tokenize_cipher_line(c_line)
        p_toks = tokenize_plain_line(p_line)

        if c_toks is None:
            num_skipped += 1
            continue

        if len(c_toks) != len(p_toks):
            num_skipped += 1
            continue

        valid = True
        for ct, pt in zip(c_toks, p_toks):
            if ct == SPACE_TOKEN and pt != " ":
                valid = False
                break
                
        if not valid:
            num_skipped += 1
            continue

        aligned_cipher_tokens.append(c_toks)
        aligned_plain_tokens.append(p_toks)

    print(f"[Preprocess] Tokenized {len(aligned_cipher_tokens)} clean pairs (skipped {num_skipped})")

    cipher_vocab = Vocab()
    cipher_vocab.build_vocab(aligned_cipher_tokens, special_tokens=["<PAD>", "<UNK>", SPACE_TOKEN])

    plain_vocab = Vocab()
    plain_vocab.build_vocab(aligned_plain_tokens, special_tokens=["<PAD>", "<UNK>"])

    # 4. Encode
    cipher_encoded, plain_encoded = build_dataset(
        aligned_cipher_tokens, aligned_plain_tokens, cipher_vocab, plain_vocab
    )

    # 5. Pad
    dataset_dict = pad_dataset(
        cipher_encoded, plain_encoded, cipher_vocab, plain_vocab
    )
    dataset = {
        "input": torch.tensor(dataset_dict["input"], dtype=torch.long),
        "target": torch.tensor(dataset_dict["target"], dtype=torch.long),
        "lengths": torch.tensor(dataset_dict["lengths"], dtype=torch.long),
        "max_len": dataset_dict["max_len"]
    }

    # 6. Save
    save_preprocessed(dataset, cipher_vocab, plain_vocab, save_dir)

    print(" Preprocessing complete and saved!")
    print(f"Cipher vocab size: {cipher_vocab.size}")
    print(f"Plain vocab size: {plain_vocab.size}")
    print(f"Max aligned length: {dataset['max_len']}")


if __name__ == "__main__":
    main()