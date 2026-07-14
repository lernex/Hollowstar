from __future__ import annotations

import argparse
from pathlib import Path

import torch


def rank_path(base_path: Path, rank: int) -> Path:
    if rank == 0:
        return base_path
    return base_path.with_name(f"{base_path.stem}.rank{rank:03d}{base_path.suffix}")


def is_grouped_expert_tensor(name: str, tensor: torch.Tensor) -> bool:
    return ".mlp.grouped_experts." in name and tensor.ndim >= 1


def merge_state_dicts(rank_states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    rank0 = rank_states[0]
    for name, tensor in rank0.items():
        if is_grouped_expert_tensor(name, tensor):
            pieces = [state[name] for state in rank_states]
            tail_shape = tuple(tensor.shape[1:])
            if any(tuple(piece.shape[1:]) != tail_shape for piece in pieces):
                raise RuntimeError(f"Expert shard shape mismatch for {name}.")
            merged[name] = torch.cat(pieces, dim=0).contiguous()
        else:
            merged[name] = tensor
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Metis-1.5 expert-parallel checkpoint shards for export/inference.")
    parser.add_argument("--checkpoint", required=True, help="Rank-0 checkpoint path, for example checkpoints/metis15_base/best.pt")
    parser.add_argument("--out", required=True, help="Output full checkpoint path.")
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    out_path = Path(args.out)
    checkpoints = []
    for rank in range(args.world_size):
        path = rank_path(checkpoint_path, rank)
        if not path.exists():
            raise FileNotFoundError(f"Missing rank {rank} checkpoint shard: {path}")
        checkpoints.append(torch.load(path, map_location="cpu"))

    model_states = [checkpoint["model_state_dict"] for checkpoint in checkpoints]
    merged = dict(checkpoints[0])
    merged["model_state_dict"] = merge_state_dicts(model_states)
    train_args = dict(merged.get("train_args") or {})
    train_args["merged_from_expert_parallel_world_size"] = args.world_size
    merged["train_args"] = train_args
    if isinstance(merged.get("model_config"), dict):
        merged["model_config"] = dict(merged["model_config"])
        merged["model_config"]["moe_expert_parallel_size"] = 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out_path)
    print(f"merged {args.world_size} expert-parallel shards -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
