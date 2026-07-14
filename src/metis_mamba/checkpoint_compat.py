from __future__ import annotations

from collections.abc import Mapping

import torch


def fuse_legacy_metis_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    consume_legacy: bool = True,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Map older unfused Metis checkpoints onto the fused QKV/SwiGLU layout."""
    converted = dict(state_dict)
    conversions: list[str] = []

    q_suffix = ".self_attn.q_proj.impl.weight"
    for q_key in list(converted):
        if not q_key.endswith(q_suffix):
            continue
        prefix = q_key[: -len("q_proj.impl.weight")]
        k_key = f"{prefix}k_proj.impl.weight"
        v_key = f"{prefix}v_proj.impl.weight"
        fused_key = f"{prefix}qkv_proj.impl.weight"
        if fused_key in converted or k_key not in converted or v_key not in converted:
            continue
        converted[fused_key] = torch.cat([converted[q_key], converted[k_key], converted[v_key]], dim=0)
        conversions.append(f"{prefix}qkv_proj.impl.weight")

        q_bias_key = f"{prefix}q_proj.impl.bias"
        k_bias_key = f"{prefix}k_proj.impl.bias"
        v_bias_key = f"{prefix}v_proj.impl.bias"
        fused_bias_key = f"{prefix}qkv_proj.impl.bias"
        if (
            fused_bias_key not in converted
            and q_bias_key in converted
            and k_bias_key in converted
            and v_bias_key in converted
        ):
            converted[fused_bias_key] = torch.cat(
                [converted[q_bias_key], converted[k_bias_key], converted[v_bias_key]],
                dim=0,
            )

        if consume_legacy:
            for key in (q_key, k_key, v_key, q_bias_key, k_bias_key, v_bias_key):
                converted.pop(key, None)

    gate_suffix = ".mlp.gate_proj.impl.weight"
    for gate_key in list(converted):
        if not gate_key.endswith(gate_suffix):
            continue
        prefix = gate_key[: -len("gate_proj.impl.weight")]
        up_key = f"{prefix}up_proj.impl.weight"
        fused_key = f"{prefix}gate_up_proj.impl.weight"
        if fused_key in converted or up_key not in converted:
            continue
        converted[fused_key] = torch.cat([converted[gate_key], converted[up_key]], dim=0)
        conversions.append(f"{prefix}gate_up_proj.impl.weight")

        gate_bias_key = f"{prefix}gate_proj.impl.bias"
        up_bias_key = f"{prefix}up_proj.impl.bias"
        fused_bias_key = f"{prefix}gate_up_proj.impl.bias"
        if fused_bias_key not in converted and gate_bias_key in converted and up_bias_key in converted:
            converted[fused_bias_key] = torch.cat([converted[gate_bias_key], converted[up_bias_key]], dim=0)

        if consume_legacy:
            for key in (gate_key, up_key, gate_bias_key, up_bias_key):
                converted.pop(key, None)

    norm_suffixes = (".ffn_norm.weight", ".ffn_norm.impl.weight")
    for norm_key in list(converted):
        matching_suffix = next((suffix for suffix in norm_suffixes if norm_key.endswith(suffix)), None)
        if matching_suffix is None:
            continue
        prefix = norm_key[: -len(matching_suffix.lstrip("."))]
        fused_norm_key = f"{prefix}mlp.impl.layer_norm_weight"
        if fused_norm_key not in converted:
            converted[fused_norm_key] = converted[norm_key]
            conversions.append(f"{prefix}mlp.impl.layer_norm_weight")

    fused_gate_suffix = ".mlp.gate_up_proj.impl.weight"
    for gate_up_key in list(converted):
        if not gate_up_key.endswith(fused_gate_suffix):
            continue
        prefix = gate_up_key[: -len("mlp.gate_up_proj.impl.weight")]
        fused_key = f"{prefix}mlp.impl.fc1_weight"
        if fused_key not in converted:
            converted[fused_key] = converted[gate_up_key]
            conversions.append(f"{prefix}mlp.impl.fc1_weight")

        gate_up_bias_key = f"{prefix}mlp.gate_up_proj.impl.bias"
        fused_bias_key = f"{prefix}mlp.impl.fc1_bias"
        if fused_bias_key not in converted and gate_up_bias_key in converted:
            converted[fused_bias_key] = converted[gate_up_bias_key]

    down_suffix = ".mlp.down_proj.impl.weight"
    for down_key in list(converted):
        if not down_key.endswith(down_suffix):
            continue
        prefix = down_key[: -len("mlp.down_proj.impl.weight")]
        fused_key = f"{prefix}mlp.impl.fc2_weight"
        if fused_key not in converted:
            converted[fused_key] = converted[down_key]
            conversions.append(f"{prefix}mlp.impl.fc2_weight")

        down_bias_key = f"{prefix}mlp.down_proj.impl.bias"
        fused_bias_key = f"{prefix}mlp.impl.fc2_bias"
        if fused_bias_key not in converted and down_bias_key in converted:
            converted[fused_bias_key] = converted[down_bias_key]

    te_norm_suffix = ".mlp.impl.layer_norm_weight"
    for norm_key in list(converted):
        if not norm_key.endswith(te_norm_suffix):
            continue
        prefix = norm_key[: -len("mlp.impl.layer_norm_weight")]
        unfused_norm_key = f"{prefix}ffn_norm.weight"
        if unfused_norm_key not in converted:
            converted[unfused_norm_key] = converted[norm_key]
            conversions.append(unfused_norm_key)

    te_fc1_suffix = ".mlp.impl.fc1_weight"
    for fc1_key in list(converted):
        if not fc1_key.endswith(te_fc1_suffix):
            continue
        prefix = fc1_key[: -len("mlp.impl.fc1_weight")]
        unfused_key = f"{prefix}mlp.gate_up_proj.impl.weight"
        if unfused_key not in converted:
            converted[unfused_key] = converted[fc1_key]
            conversions.append(unfused_key)

        fc1_bias_key = f"{prefix}mlp.impl.fc1_bias"
        unfused_bias_key = f"{prefix}mlp.gate_up_proj.impl.bias"
        if unfused_bias_key not in converted and fc1_bias_key in converted:
            converted[unfused_bias_key] = converted[fc1_bias_key]

    te_fc2_suffix = ".mlp.impl.fc2_weight"
    for fc2_key in list(converted):
        if not fc2_key.endswith(te_fc2_suffix):
            continue
        prefix = fc2_key[: -len("mlp.impl.fc2_weight")]
        unfused_key = f"{prefix}mlp.down_proj.impl.weight"
        if unfused_key not in converted:
            converted[unfused_key] = converted[fc2_key]
            conversions.append(unfused_key)

        fc2_bias_key = f"{prefix}mlp.impl.fc2_bias"
        unfused_bias_key = f"{prefix}mlp.down_proj.impl.bias"
        if unfused_bias_key not in converted and fc2_bias_key in converted:
            converted[unfused_bias_key] = converted[fc2_bias_key]

    return converted, conversions


def filter_state_dict_for_model(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    converted, conversions = fuse_legacy_metis_state_dict(state_dict)
    target_state = model.state_dict()
    expert_parallel_prefixes: list[tuple[str, int, int]] = []
    for module_name, module in model.named_modules():
        if not bool(getattr(module, "expert_parallel_enabled", False)):
            continue
        start = int(getattr(module, "local_expert_start"))
        end = int(getattr(module, "local_expert_end"))
        prefix = f"{module_name}.grouped_experts." if module_name else "grouped_experts."
        expert_parallel_prefixes.append((prefix, start, end))

    filtered: dict[str, torch.Tensor] = {}
    for name, tensor in converted.items():
        target = target_state.get(name)
        if target is None:
            continue
        if tuple(tensor.shape) == tuple(target.shape):
            filtered[name] = tensor
            continue
        for prefix, start, end in expert_parallel_prefixes:
            if not name.startswith(prefix):
                continue
            if tensor.ndim == 0 or tensor.shape[0] < end:
                continue
            sharded = tensor.narrow(0, start, end - start).contiguous()
            if tuple(sharded.shape) == tuple(target.shape):
                filtered[name] = sharded
                conversions.append(f"{name}[experts {start}:{end}]")
            break
    return filtered, conversions
