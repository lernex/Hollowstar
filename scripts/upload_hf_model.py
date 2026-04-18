from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from huggingface_hub import HfApi


def build_readme(repo_id: str, checkpoint_path: Path, tokenizer_path: Path | None) -> str:
    lines = [
        f"# {repo_id}",
        "",
        "Private Metis checkpoint uploaded from the local training workspace.",
        "",
        "## Included files",
        "",
        f"- checkpoint: `{checkpoint_path.name}`",
    ]
    if tokenizer_path is not None:
        lines.append(f"- tokenizer: `{tokenizer_path.name}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a Metis checkpoint and tokenizer files to a Hugging Face model repo.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--create-repo", action="store_true")
    parser.add_argument("--message", default="Upload Metis checkpoint")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required in the environment.")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path else None
    if tokenizer_path is not None and not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    api = HfApi(token=token)
    if args.create_repo:
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    uploads: list[tuple[Path, str]] = [(checkpoint_path, checkpoint_path.name)]

    train_config = checkpoint_path.with_name("train_config.json")
    if train_config.exists():
        uploads.append((train_config, "train_config.json"))

    if tokenizer_path is not None:
        uploads.append((tokenizer_path, tokenizer_path.name))
        tokenizer_meta = tokenizer_path.with_name("tokenizer_meta.json")
        if tokenizer_meta.exists():
            uploads.append((tokenizer_meta, "tokenizer_meta.json"))

    readme_text = build_readme(args.repo_id, checkpoint_path, tokenizer_path)
    with NamedTemporaryFile("w", suffix=".md", delete=False) as handle:
        handle.write(readme_text)
        readme_path = Path(handle.name)
    uploads.append((readme_path, "README.md"))

    try:
        for local_path, repo_path in uploads:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=args.message,
            )
    finally:
        if readme_path.exists():
            readme_path.unlink()

    summary = {
        "repo_id": args.repo_id,
        "checkpoint": str(checkpoint_path),
        "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
        "uploaded_files": [repo_path for _, repo_path in uploads],
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
