import os
import torch
import pickle

def save_cache(tensors, vocab, save_dir: str):
    """
    Save the encoded + padded sequences, and vocabulary.
    Do NOT cache MLM masking per requirements.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(tensors, os.path.join(save_dir, "dataset_task2.pt"))
    
    with open(os.path.join(save_dir, "vocab_task2.pkl"), "wb") as f:
        pickle.dump(vocab, f)

def load_cache(save_dir: str):
    tensors = torch.load(os.path.join(save_dir, "dataset_task2.pt"), weights_only=True)
    with open(os.path.join(save_dir, "vocab_task2.pkl"), "rb") as f:
        vocab = pickle.load(f)
    return tensors, vocab
