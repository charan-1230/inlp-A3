import torch
import pickle
import os


def save_preprocessed(data, cipher_vocab, plain_vocab, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    torch.save(data, os.path.join(save_dir, "dataset.pt"))

    with open(os.path.join(save_dir, "cipher_vocab.pkl"), "wb") as f:
        pickle.dump(cipher_vocab, f)

    with open(os.path.join(save_dir, "plain_vocab.pkl"), "wb") as f:
        pickle.dump(plain_vocab, f)


def load_preprocessed(save_dir):
    data = torch.load(os.path.join(save_dir, "dataset.pt"))

    with open(os.path.join(save_dir, "cipher_vocab.pkl"), "rb") as f:
        cipher_vocab = pickle.load(f)

    with open(os.path.join(save_dir, "plain_vocab.pkl"), "rb") as f:
        plain_vocab = pickle.load(f)

    return data, cipher_vocab, plain_vocab