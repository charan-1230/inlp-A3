import os

import torch
import yaml

from src.task1.dataset import load_datasets, load_test_file
from src.task1.model_lstm import SeqLabelLSTM
from src.task1.trainer import Trainer, indices_to_string
from src.task1.metrics import compute_metrics, save_metrics
from src.utils.checkpoints import load_checkpoint
from src.utils.hf_wandb import init_wandb, finish_wandb


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(config: dict, meta: dict, device) -> SeqLabelLSTM:
    model = SeqLabelLSTM(
        config=config,
        cipher_vocab_size=meta["cipher_vocab_size"],
        plain_vocab_size=meta["plain_vocab_size"],
        pad_idx=meta.get("pad_idx", 0),
    )
    return model.to(device)


def _checkpoint_path(config: dict) -> str:
    return config.get("checkpoint_path")


def _output_dir(config: dict) -> str:
    return config.get("output_dir", os.path.join("outputs", "results"))


def train(config: dict):
    print("[task1_lstm] Starting training...")

    train_loader, val_loader, meta = load_datasets(config)
    device = _get_device()

    plain_vocab = meta["plain_vocab"]
    meta["pad_idx"] = plain_vocab.char2idx['<PAD>']

    model = _build_model(config, meta, device)

    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        init_wandb(
            project=config.get("wandb_project", "inlp-a3-task1"),
            config=config,
            name="task1_lstm",
        )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        meta=meta,
        checkpoint_path=_checkpoint_path(config),
        use_wandb=use_wandb,
    )

    trainer.train()

    if use_wandb:
        finish_wandb()

    print("[task1_lstm] Training done.")


def evaluate(config: dict):
    print("[task1_lstm] Starting evaluation...")

    _, val_loader, meta = load_datasets(config)
    device = _get_device()

    plain_vocab  = meta["plain_vocab"]
    cipher_vocab = meta["cipher_vocab"]
    pad_idx      = plain_vocab.char2idx['<PAD>']
    
    meta["pad_idx"] = pad_idx
    idx2plain    = {v: k for k, v in plain_vocab.items()}

    model = _build_model(config, meta, device)

    ckpt_path = _checkpoint_path(config)
    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"[task1_lstm] Loading local checkpoint: {ckpt_path}")
        load_checkpoint(ckpt_path, model, device=str(device))
    elif config.get("hf_repo_id"):
        print(f"[task1_lstm] Local checkpoint not found at '{ckpt_path}'.")
        print(f"[task1_lstm] Attempting to pull from HuggingFace: {config['hf_repo_id']}")
        from src.utils.hf_wandb import pull_from_hub
        local_dir = os.path.dirname(ckpt_path) if ckpt_path else os.path.join("outputs", "checkpoints")
        filename  = os.path.basename(ckpt_path) if ckpt_path else "lstm_best.pt"
        try:
            pulled = pull_from_hub(config["hf_repo_id"], filename, local_dir=local_dir)
            load_checkpoint(pulled, model, device=str(device))
            print(f"[task1_lstm] Loaded checkpoint from HuggingFace.")
        except Exception as e:
            raise FileNotFoundError(
                f"\n\n[task1_lstm] Could not load checkpoint.\n"
                f"  Local path : {ckpt_path}  → not found\n"
                f"  HuggingFace: {config['hf_repo_id']}/{filename}  → {e}\n"
                f"\n  To fix: train the model first with --mode train\n"
            ) from None
    else:
        raise FileNotFoundError(
            f"[task1_lstm] No checkpoint found at '{ckpt_path}' and no hf_repo_id configured.\n"
            f"  Run with --mode train first."
        )
    model.eval()

    out_dir = _output_dir(config)
    os.makedirs(out_dir, exist_ok=True)

    max_plain_len = meta.get("max_len", 200)

    all_preds, all_targets = [], []

    with torch.no_grad():
        for cipher, plain, lengths in val_loader:
            cipher = cipher.to(device)
            lengths  = lengths.to(device)

            logits = model(cipher, lengths)
            preds  = logits.argmax(dim=-1)

            for i in range(plain.shape[0]):
                act_len = lengths[i].item()
                tgt_str  = indices_to_string(plain[i][:act_len], idx2plain, pad_idx)
                pred_str = indices_to_string(preds[i][:act_len], idx2plain, pad_idx)
                all_targets.append(tgt_str)
                all_preds.append(pred_str)

    metrics = compute_metrics(all_preds, all_targets)
    print("[task1_lstm] Validation metrics:", metrics)
    save_metrics(metrics, out_dir, filename="lstm_metrics.json")

    # ---- External test file ----
    test_file = config.get("test_file", None)
    if test_file and os.path.isfile(test_file):
        print(f"[task1_lstm] Running inference on test file: {test_file}")
        tensors, lengths, raw_lines = load_test_file(
            test_file, cipher_vocab, max_plain_len, str(device)
        )

        batch_size = config.get("batch_size", 32)
        test_preds = []
        with torch.no_grad():
            for start in range(0, len(tensors), batch_size):
                batch_c = tensors[start : start + batch_size]
                batch_l = lengths[start : start + batch_size]
                
                logits = model(batch_c, batch_l)
                preds  = logits.argmax(dim=-1)

                for idx, row in enumerate(preds):
                    actual_length = batch_l[idx].item()
                    test_preds.append(indices_to_string(row[:actual_length], idx2plain, pad_idx))

        pred_path = os.path.join(out_dir, "task1_lstm.txt")
        with open(pred_path, "w", encoding="utf-8") as f:
            for line in test_preds:
                f.write(line + "\n")
        print(f"[task1_lstm] Predictions written to {pred_path}")
    else:
        pred_path = os.path.join(out_dir, "task1_lstm.txt")
        samples = list(zip(all_preds[:5], all_targets[:5]))
        with open(pred_path, "w", encoding="utf-8") as f:
            for pred, tgt in samples:
                f.write(f"GT  : {tgt}\n")
                f.write(f"PRED: {pred}\n")
                f.write("-" * 60 + "\n")
        print(f"[task1_lstm] 5 sample predictions written to {pred_path}")

    print("[task1_lstm] Evaluation done.")


def main(config_path: str, mode: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if mode == "train":
        train(config)
    elif mode == "evaluate":
        evaluate(config)
    elif mode == "both":
        train(config)
        evaluate(config)
    else:
        raise ValueError(f"Unsupported mode: {mode}")