#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm


class DenseBench(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, hidden_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.up = nn.Linear(d_model, hidden_size, bias=False)
        self.down = nn.Linear(hidden_size, d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.down(F.silu(self.up(x)))
        return self.lm_head(x)


class LocalExpertBench(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, local_experts: int) -> None:
        super().__init__()
        self.up = nn.Parameter(torch.empty(local_experts, d_model, hidden_size))
        self.down = nn.Parameter(torch.empty(local_experts, hidden_size, d_model))
        nn.init.normal_(self.up, mean=0.0, std=0.02)
        nn.init.normal_(self.down, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, *, mode: str) -> torch.Tensor:
        if mode == "all":
            out = None
            for idx in range(self.up.shape[0]):
                hidden = F.silu(torch.matmul(tokens, self.up[idx]))
                expert_out = torch.matmul(hidden, self.down[idx])
                out = expert_out if out is None else out + expert_out
            return out

        # Static expert-local compute: no token routing, one fixed token slice per local expert.
        chunks = tokens.chunk(self.up.shape[0], dim=1)
        outs = []
        for idx, chunk in enumerate(chunks):
            hidden = F.silu(torch.matmul(chunk, self.up[idx]))
            outs.append(torch.matmul(hidden, self.down[idx]))
        return torch.cat(outs, dim=1)


def maybe_init_distributed() -> tuple[bool, int, int]:
    if "RANK" not in os.environ:
        return False, 0, 1
    import torch_xla.distributed.xla_backend  # noqa: F401

    dist.init_process_group("xla")
    return True, dist.get_rank(), dist.get_world_size()


def bench_dense(args: argparse.Namespace, device: torch.device, distributed: bool) -> dict[str, float | int | str]:
    torch.manual_seed(args.seed)
    model = DenseBench(args.vocab_size, args.d_model, args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
    labels = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

    start = None
    for step in range(args.warmup_steps + args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(dtype=torch.bfloat16, device_type="xla"):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), labels.reshape(-1))
        loss.backward()
        if distributed:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
            xm.mark_step()
        if step == args.warmup_steps - 1:
            xm.wait_device_ops()
            start = time.perf_counter()

    xm.wait_device_ops()
    elapsed = time.perf_counter() - (start or time.perf_counter())
    tokens = args.batch_size * args.seq_len * args.steps
    return {
        "bench": "dense_train",
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "hidden_size": args.hidden_size,
        "vocab_size": args.vocab_size,
        "steps": args.steps,
        "elapsed_s": elapsed,
        "tokens_per_s_per_rank": tokens / elapsed,
    }


def bench_local_expert(args: argparse.Namespace, device: torch.device, distributed: bool) -> dict[str, float | int | str]:
    torch.manual_seed(args.seed)
    model = LocalExpertBench(args.d_model, args.hidden_size, args.local_experts).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tokens = torch.randn(args.batch_size, args.seq_len, args.d_model, device=device)
    target = torch.randn(args.batch_size, args.seq_len, args.d_model, device=device)

    start = None
    for step in range(args.warmup_steps + args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(dtype=torch.bfloat16, device_type="xla"):
            out = model(tokens, mode=args.expert_mode)
            loss = F.mse_loss(out.float(), target.float())
        loss.backward()
        if distributed and args.sync_expert_grads:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
            xm.mark_step()
        if step == args.warmup_steps - 1:
            xm.wait_device_ops()
            start = time.perf_counter()

    xm.wait_device_ops()
    elapsed = time.perf_counter() - (start or time.perf_counter())
    tokens_count = args.batch_size * args.seq_len * args.steps
    experts_per_token = args.local_experts if args.expert_mode == "all" else 1
    ffn_flops = 4 * tokens_count * args.d_model * args.hidden_size * experts_per_token
    return {
        "bench": "local_expert_train",
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "hidden_size": args.hidden_size,
        "local_experts": args.local_experts,
        "expert_mode": args.expert_mode,
        "experts_per_token": experts_per_token,
        "steps": args.steps,
        "elapsed_s": elapsed,
        "tokens_per_s_per_rank": tokens_count / elapsed,
        "ffn_tflops_per_rank": ffn_flops / elapsed / 1e12,
        "expert_grads_synced": int(args.sync_expert_grads),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-shape PyTorch/XLA Neuron throughput probes.")
    parser.add_argument("--bench", choices=["dense", "local-expert"], default="dense")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--hidden-size", type=int, default=3072)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--local-experts", type=int, default=4)
    parser.add_argument("--expert-mode", choices=["chunk", "all"], default="chunk")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sync-expert-grads", action="store_true")
    args = parser.parse_args()

    distributed, rank, world_size = maybe_init_distributed()
    device = xm.xla_device()
    if args.bench == "dense":
        result = bench_dense(args, device, distributed)
    else:
        result = bench_local_expert(args, device, distributed)

    result["rank"] = rank
    result["world_size"] = world_size
    result["xla_device_kind"] = xm.xla_device_kind()
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
