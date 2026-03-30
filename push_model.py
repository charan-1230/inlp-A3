"""
push_model.py — Push a saved checkpoint file to HuggingFace Hub on demand.

Usage:
    uv run push_model.py --checkpoint outputs/checkpoints/task1_lstm.pt \
                         --repo-id your-username/inlp-a3 \
                         [--path-in-repo task1_lstm.pt]

The HF_TOKEN_PUSH env var (or .env file) is used for authentication.
You can also pass --token explicitly.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Push a local checkpoint file to HuggingFace Hub."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the local checkpoint file (e.g. outputs/checkpoints/task1_lstm.pt)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID to push to (e.g. your-username/inlp-a3)",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        default=None,
        help="Filename inside the repo. Defaults to the basename of --checkpoint.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token. Falls back to HF_TOKEN env var.",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if not os.path.isfile(checkpoint_path):
        print(f"[push_model] ERROR: checkpoint not found at '{checkpoint_path}'")
        sys.exit(1)

    from datetime import datetime

    if args.path_in_repo:
        path_in_repo = args.path_in_repo
    else:
        # Default is the exact filename (e.g. task1_rnn.pt). 
        # HuggingFace uses Git, so this automatically versions your files without overwriting history!
        path_in_repo = os.path.basename(checkpoint_path)

    # Import here so it's easy to run standalone
    from src.utils.hf_wandb import push_to_hub

    print(f"[push_model] Pushing '{checkpoint_path}' → '{args.repo_id}/{path_in_repo}' ...")
    url = push_to_hub(
        path=checkpoint_path,
        repo_id=args.repo_id,
        path_in_repo=path_in_repo,
        token=args.token,
    )
    print(f"[push_model] Done! Uploaded to: {url}")


if __name__ == "__main__":
    main()
