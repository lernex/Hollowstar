from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ASSET_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def build_readme(manifest: dict, stage_name: str, repo_id: str | None) -> str:
    title = repo_id or f"{manifest['name']}-{stage_name}"
    lines = [
        f"# {title}",
        "",
        f"{manifest['name']} {stage_name} release artifact.",
        "",
        "## Files",
        "",
        "- `model.safetensors`",
        "- `config.json`",
        "- `generation_config.json`",
        "- `tokenizer.json`",
        "- `tokenizer_config.json`",
        "- `special_tokens_map.json`",
        "",
        "## Notes",
        "",
        "This folder is assembled from the shared HF assets plus the exported BF16 model weights.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a clean HF release directory from local export artifacts.")
    parser.add_argument("--manifest", default="configs/metis12_manifest.json")
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repo-id", default=None)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    assets_dir = Path(args.assets_dir)
    export_dir = Path(args.export_dir)
    out_dir = Path(args.out_dir)
    tmp_dir = out_dir.with_name(f"{out_dir.name}.tmp")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for filename in ASSET_FILES:
        src = assets_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing HF asset: {src}")
        shutil.copy2(src, tmp_dir / filename)

    copied_model_files: list[str] = []
    for candidate in export_dir.iterdir():
        if candidate.name.startswith("model.safetensors"):
            shutil.copy2(candidate, tmp_dir / candidate.name)
            copied_model_files.append(candidate.name)

    if not copied_model_files:
        raise FileNotFoundError(f"No exported model.safetensors files found in {export_dir}")

    readme = build_readme(manifest, args.stage_name, args.repo_id)
    (tmp_dir / "README.md").write_text(readme)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.replace(out_dir)

    summary = {
        "stage_name": args.stage_name,
        "out_dir": str(out_dir),
        "model_files": copied_model_files,
        "repo_id": args.repo_id,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
