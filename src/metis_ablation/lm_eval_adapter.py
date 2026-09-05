"""Native, non-chat evaluation of the completed fixed-depth dense control.

The harness is an optional dependency: ``as_harness_lm`` creates its real LM
subclass, while the numerical adapter can be tested without installing it.
Generation deliberately recomputes the full prefix; Metis16 has no incremental
inference API. No method crops prompts, answers, or generated prefixes.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from metis_training.model import CurriculumState


class ContextOverflowError(ValueError):
    """The requested evaluation cannot be performed without losing context."""


class TokenBoundaryError(ValueError):
    """The requested text continuation is not aligned with a token boundary."""


def fixed_dense_curriculum(config: Any, saved: Mapping[str, Any] | CurriculumState):
    state = CurriculumState.from_value(saved)
    if (
        config.ffn_mode != "dense"
        or config.n_routed_experts != 0
        or config.n_shared_experts != 0
        or state.continuation_mode != "depth_one"
        or state.memory_gate_scale != 0.0
        or state.ngram_gate_scale != 1.0
    ):
        raise ValueError("Evaluation requires the saved depth_one, memory-off dense control")
    if getattr(state, "compute_allocation_mode", "legacy") != "legacy":
        raise ValueError("The dense control must not use a joint/adaptive compute policy")
    state = replace(state, stochastic_routing=False)
    state.validate(config)
    return state


def rolling_windows(
    tokens: Sequence[int], *, max_length: int, prefix_token_id: int,
) -> Iterable[tuple[list[int], list[int]]]:
    """Harness rolling semantics: every target once, full final-window context."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    for start in range(0, len(tokens), max_length):
        end = min(start + max_length, len(tokens))
        context = (
            [prefix_token_id] if start == 0
            else list(tokens[max(0, end - max_length - 1):start])
        )
        yield context, list(tokens[start:end])


@dataclass(frozen=True)
class GenerationOptions:
    until: tuple[str, ...]
    max_new_tokens: int


class NativeDenseLM:
    """Row-preserving native inference, with explicit continuation-only scoring."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        curriculum: Mapping[str, Any] | CurriculumState,
        canonical_id_lookup: Sequence[int] | Tensor,
        eos_token_id: int,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        batch_size: int = 1,
        max_length: int | None = None,
        max_gen_toks: int = 256,
        logit_chunk_size: int = 128,
        reuse_single_token_contexts: bool = True,
        request_validator: Callable[[Sequence[Any]], None] | None = None,
        progress_callback: Callable[[Mapping[str, int | float]], None] | None = None,
        device: str | torch.device = "cpu",
        precision_context: Callable[..., Any] = nullcontext,
    ) -> None:
        self.model = model.eval()
        self.config = model.config
        self.curriculum = fixed_dense_curriculum(self.config, curriculum)
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.precision_context = precision_context
        self.batch_size = batch_size
        self.max_length = self.config.final_context_length if max_length is None else max_length
        self.max_gen_toks = max_gen_toks
        self.logit_chunk_size = logit_chunk_size
        self.reuse_single_token_contexts = reuse_single_token_contexts
        self.request_validator = request_validator
        self.progress_callback = progress_callback
        for name in ("batch_size", "max_length", "max_gen_toks", "logit_chunk_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_length > self.config.final_context_length:
            raise ValueError("Adapter context bound exceeds the native model context bound")
        if (
            self.config.expert_parallel_size != 1
            or self.config.context_parallel_size != 1
            or self.config.ngram_memory.table_mode != "replicated"
        ):
            raise ValueError("Evaluation requires replicated weights and whole sequence rows")
        self.eot_token_id = eos_token_id
        self.pad_token_id = eos_token_id if pad_token_id is None else pad_token_id
        self.bos_token_id = bos_token_id
        self.prefix_token_id = eos_token_id if bos_token_id is None else bos_token_id
        for token in (self.eot_token_id, self.pad_token_id, self.prefix_token_id):
            if type(token) is not int or not 0 <= token < self.config.vocab_size:
                raise ValueError("Special token ID is outside the model vocabulary")
        if tokenizer.get_vocab_size(with_added_tokens=True) != self.config.vocab_size:
            raise ValueError("Tokenizer and model vocabulary sizes differ")
        if getattr(tokenizer, "truncation", None) is not None:
            raise ValueError("Tokenizer-level truncation must be disabled")
        if getattr(tokenizer, "padding", None) is not None:
            raise ValueError("Tokenizer-level padding must be disabled")
        lookup = torch.as_tensor(canonical_id_lookup, device=self.device)
        if lookup.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint16):
            raise ValueError("Canonical-ID lookup must contain integers")
        lookup = lookup.long()
        if (
            lookup.shape != (self.config.vocab_size,)
            or bool((lookup < 0).any())
            or bool((lookup >= self.config.vocab_size).any())
        ):
            raise ValueError("Canonical-ID lookup must cover the complete tokenizer vocabulary")
        self.canonical_id_lookup = lookup
        self.stats: dict[str, int | float] = {
            "forward_calls": 0, "forward_input_tokens": 0, "forward_padded_tokens": 0,
            "max_forward_length": 0, "model_seconds": 0.0,
            "loglikelihood_requests": 0, "rolling_requests": 0,
            "scored_tokens": 0, "generation_requests": 0, "generated_tokens": 0,
            "generation_eos_stops": 0, "generation_text_stops": 0,
            "generation_token_limit_stops": 0,
            "generation_possible_context_overflows": 0,
            "boundary_errors": 0, "context_overflows": 0,
            "single_token_context_reuse_hits": 0,
            "completed_likelihood_pairs": 0,
        }
        self._request_hashes: dict[str, Any] = {}

    def _record_requests(self, method: str, requests: Sequence[Any]) -> None:
        digest = self._request_hashes.setdefault(method, hashlib.sha256())
        for request in requests:
            data = json.dumps(request.args, ensure_ascii=False, separators=(",", ":"),
                              sort_keys=True).encode("utf-8")
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)

    def metadata(self) -> dict[str, Any]:
        return {
            "stats": dict(self.stats),
            "request_sha256": {name: digest.hexdigest() for name, digest in self._request_hashes.items()},
            "generation_backend": "full_prefix_no_cache",
            "generation_cost": "G forwards; P*G + G*(G-1)/2 useful input tokens per unstopped request",
            "chat_template": None,
            "padding_side": "right",
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eot_token_id,
            "pad_token_id": self.pad_token_id,
            "empty_context_prefix_token_id": self.prefix_token_id,
            "add_special_tokens": False,
            "token_boundary_policy": "joint encoding; move trailing prompt whitespace; reject unstable prefix",
            "overflow_policy": "error_without_truncation",
            "max_context": self.max_length,
            "default_max_gen_toks": self.max_gen_toks,
            "logit_chunk_size": self.logit_chunk_size,
            "logit_projection": "continuation positions only; native final hidden state and LM head",
            "reuse_single_token_contexts": self.reuse_single_token_contexts,
        }

    def tok_encode(self, text: str, **kwargs: Any) -> list[int]:
        if kwargs:
            raise ValueError("Tokenization overrides are not supported")
        if not isinstance(text, str):
            raise TypeError("Evaluation inputs must be strings")
        tokens = list(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if any(type(token) is not int or not 0 <= token < self.config.vocab_size for token in tokens):
            raise ValueError("Tokenizer emitted a token outside the model vocabulary")
        return tokens

    def tok_decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(list(tokens), skip_special_tokens=False)

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        spaces = len(context) - len(context.rstrip())
        if spaces:
            continuation = context[-spaces:] + continuation
            context = context[:-spaces]
        prefix = self.tok_encode(context)
        joined = self.tok_encode(context + continuation)
        # Slicing only by len(prefix) silently drops a boundary-spanning BPE
        # token. Re-scoring that token instead changes the conditioning event.
        if joined[:len(prefix)] != prefix:
            self.stats["boundary_errors"] += 1
            raise TokenBoundaryError(
                "Joint tokenization changes the prompt prefix; the continuation is not "
                "token-aligned. Refusing to drop or re-score prompt tokens."
            )
        targets = joined[len(prefix):]
        if continuation and not targets:
            raise TokenBoundaryError("A nonempty continuation encoded to no target tokens")
        if self.bos_token_id is not None:
            prefix = [self.bos_token_id] + prefix
        return prefix or [self.prefix_token_id], targets

    def _check_length(self, length: int, *, operation: str, index: int) -> None:
        if length > self.max_length:
            self.stats["context_overflows"] += 1
            raise ContextOverflowError(
                f"{operation} request {index} requires {length} input tokens; "
                f"native context bound is {self.max_length}. Nothing was truncated."
            )

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def _hidden(self, rows: Sequence[Sequence[int]]) -> Tensor:
        lengths = [len(row) for row in rows]
        if not lengths or min(lengths) < 1 or max(lengths) > self.max_length:
            raise ValueError("Forward rows must be nonempty and within the context bound")
        width = max(lengths)
        ids = torch.full((len(rows), width), self.pad_token_id,
                         dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for index, row in enumerate(rows):
            ids[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=self.device)
            mask[index, :len(row)] = True
        documents = torch.arange(len(rows), dtype=torch.int32, device=self.device)[:, None].expand_as(ids)
        documents = documents.masked_fill(~mask, -1)
        resets = torch.zeros_like(mask)
        resets[:, 0] = True
        if width > 1:
            resets[:, 1:] |= documents[:, 1:] != documents[:, :-1]
        self.model.eval()
        self._synchronize()
        started = time.perf_counter()
        with self.precision_context():
            output = self.model(
                ids, attention_mask=mask, document_ids=documents, reset_mask=resets,
                canonical_ids=self.canonical_id_lookup[ids], curriculum=self.curriculum,
                max_passes=1, force_depth=1, return_logits=False,
            )
        if not torch.equal(output.chosen_depths, mask.long()):
            raise RuntimeError("Dense evaluation did not execute exactly one pass per valid token")
        self._synchronize()
        self.stats["model_seconds"] += time.perf_counter() - started
        self.stats["forward_calls"] += 1
        self.stats["forward_input_tokens"] += sum(lengths)
        self.stats["forward_padded_tokens"] += ids.numel() - sum(lengths)
        self.stats["max_forward_length"] = max(self.stats["max_forward_length"], width)
        return output.final_hidden_state

    @torch.no_grad()
    def _logits(self, hidden: Tensor) -> Tensor:
        with self.precision_context():
            logits = self.model._precision_call(self.model.lm_head, hidden)
        if logits.shape != (*hidden.shape[:-1], self.config.vocab_size):
            raise RuntimeError("Native LM head returned unexpected vocabulary geometry")
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("Native model produced nonfinite logits")
        return logits.float()

    @torch.no_grad()
    def _score_pairs(
        self, pairs: Sequence[tuple[list[int], list[int]]],
    ) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = [(0.0, True)] * len(pairs)
        groups: list[list[int]] = []
        single_token_contexts: dict[tuple[int, ...], list[int]] = {}
        for index, (context, target) in enumerate(pairs):
            if target:
                self._check_length(len(context) + len(target) - 1, operation="loglikelihood", index=index)
                if self.reuse_single_token_contexts and len(target) == 1:
                    key = tuple(context)
                    if key not in single_token_contexts:
                        single_token_contexts[key] = []
                        groups.append(single_token_contexts[key])
                    single_token_contexts[key].append(index)
                else:
                    groups.append([index])
        # Similar lengths reduce padding without combining independent rows.
        groups.sort(key=lambda group: len(pairs[group[0]][0]) + len(pairs[group[0]][1]), reverse=True)
        for offset in range(0, len(groups), self.batch_size):
            batch = groups[offset:offset + self.batch_size]
            indices = [group[0] for group in batch]
            hidden = self._hidden([(pairs[i][0] + pairs[i][1])[:-1] for i in indices])
            for row, (index, group) in enumerate(zip(indices, batch, strict=True)):
                context, target = pairs[index]
                start = len(context) - 1
                if len(group) > 1:
                    logits = self._logits(hidden[row, start:start + 1])[0]
                    targets = torch.tensor([pairs[i][1][0] for i in group],
                                           dtype=torch.long, device=self.device)
                    scores = F.log_softmax(logits, dim=-1)[targets].tolist()
                    greedy = int(logits.argmax().item())
                    for member, score in zip(group, scores, strict=True):
                        results[member] = (score, pairs[member][1][0] == greedy)
                    self.stats["scored_tokens"] += len(group)
                    self.stats["single_token_context_reuse_hits"] += len(group) - 1
                    continue
                logprob, greedy = 0.0, True
                for pos in range(0, len(target), self.logit_chunk_size):
                    stop = min(pos + self.logit_chunk_size, len(target))
                    logits = self._logits(hidden[row, start + pos:start + stop])
                    expected = torch.tensor(target[pos:stop], dtype=torch.long, device=self.device)
                    selected = F.log_softmax(logits, dim=-1).gather(-1, expected[:, None])
                    logprob += float(selected.sum(dtype=torch.float64).item())
                    greedy = greedy and bool(logits.argmax(-1).eq(expected).all().item())
                results[index] = (logprob, greedy)
                self.stats["scored_tokens"] += len(target)
            self.stats["completed_likelihood_pairs"] += sum(len(group) for group in batch)
            if self.progress_callback is not None:
                self.progress_callback(dict(self.stats))
        return results

    def loglikelihood(self, requests: Sequence[Any], disable_tqdm: bool = False):
        del disable_tqdm
        if self.request_validator is not None:
            self.request_validator(requests)
        self._record_requests("loglikelihood", requests)
        self.stats["loglikelihood_requests"] += len(requests)
        pairs = [
            self._encode_pair(context, continuation) if continuation else ([self.prefix_token_id], [])
            for context, continuation in (request.args for request in requests)
        ]
        return self._score_pairs(pairs)

    def loglikelihood_rolling(self, requests: Sequence[Any], disable_tqdm: bool = False):
        del disable_tqdm
        self._record_requests("loglikelihood_rolling", requests)
        self.stats["rolling_requests"] += len(requests)
        pairs, owners = [], []
        for index, request in enumerate(requests):
            (text,) = request.args
            windows = list(rolling_windows(
                self.tok_encode(text), max_length=self.max_length,
                prefix_token_id=self.prefix_token_id,
            ))
            pairs.extend(windows)
            owners.extend([index] * len(windows))
        results = [0.0] * len(requests)
        for owner, (score, _greedy) in zip(owners, self._score_pairs(pairs), strict=True):
            results[owner] += score
        return results

    def _generation_options(self, kwargs: Mapping[str, Any]) -> GenerationOptions:
        allowed = {"until", "max_gen_toks", "max_new_tokens", "do_sample", "temperature"}
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(f"Unsupported generation options: {sorted(unknown)}")
        if kwargs.get("do_sample", False) is not False:
            raise ValueError("This adapter supports deterministic greedy generation only")
        temperature = kwargs.get("temperature", 0.0)
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Greedy temperature must be a finite nonnegative number")
        if "max_gen_toks" in kwargs and "max_new_tokens" in kwargs:
            raise ValueError("Specify max_gen_toks or max_new_tokens, not both")
        maximum = kwargs.get("max_gen_toks", kwargs.get("max_new_tokens", self.max_gen_toks))
        if type(maximum) is not int or maximum < 1:
            raise ValueError("Generation token budget must be a positive integer")
        until = kwargs.get("until", ())
        if until is None:
            until = ()
        if isinstance(until, str):
            until = (until,)
        if not isinstance(until, (list, tuple)) or any(not isinstance(stop, str) for stop in until):
            raise ValueError("until must be a string or a sequence of strings")
        return GenerationOptions(tuple(until), maximum)

    @torch.no_grad()
    def generate_until(self, requests: Sequence[Any], disable_tqdm: bool = False):
        del disable_tqdm
        self._record_requests("generate_until", requests)
        self.stats["generation_requests"] += len(requests)
        prompts, options = [], []
        for index, request in enumerate(requests):
            text, kwargs = request.args
            tokens = self.tok_encode(text)
            if self.bos_token_id is not None:
                tokens = [self.bos_token_id] + tokens
            tokens = tokens or [self.prefix_token_id]
            self._check_length(len(tokens), operation="generation prompt", index=index)
            opt = self._generation_options(kwargs)
            if len(tokens) + opt.max_new_tokens - 1 > self.max_length:
                self.stats["generation_possible_context_overflows"] += 1
            prompts.append(tokens)
            options.append(opt)
        results = [""] * len(requests)
        for offset in range(0, len(requests), self.batch_size):
            indices = list(range(offset, min(offset + self.batch_size, len(requests))))
            generated: dict[int, list[int]] = {index: [] for index in indices}
            active = []
            for index in indices:
                if "" in options[index].until:
                    self.stats["generation_text_stops"] += 1
                else:
                    active.append(index)
            while active:
                rows = [prompts[index] + generated[index] for index in active]
                for index, row in zip(active, rows, strict=True):
                    self._check_length(len(row), operation="generation prefix", index=index)
                hidden = self._hidden(rows)
                last = torch.stack([hidden[row, len(tokens) - 1] for row, tokens in enumerate(rows)])
                next_ids = self._logits(last).argmax(-1).tolist()
                remaining = []
                for index, token in zip(active, next_ids, strict=True):
                    self.stats["generated_tokens"] += 1
                    if token == self.eot_token_id:
                        self.stats["generation_eos_stops"] += 1
                        continue
                    generated[index].append(token)
                    text = self.tok_decode(generated[index])
                    stops = [text.find(stop) for stop in options[index].until if stop in text]
                    if stops:
                        results[index] = text[:min(stops)]
                        self.stats["generation_text_stops"] += 1
                    else:
                        results[index] = text
                        if len(generated[index]) >= options[index].max_new_tokens:
                            self.stats["generation_token_limit_stops"] += 1
                        else:
                            remaining.append(index)
                active = remaining
        return results


def as_harness_lm(native: NativeDenseLM):
    """Wrap the tested core in the installed harness's actual abstract base."""
    from lm_eval.api.model import LM

    class MetisDenseHarnessLM(LM):
        def __init__(self):
            super().__init__()
            self.native = native
            self.tokenizer = native.tokenizer
            self.batch_size = native.batch_size
            self._device = native.device
            self.config = native.config
            self.max_length = native.max_length
            self.max_gen_toks = native.max_gen_toks

        @property
        def tokenizer_name(self):
            return "metis-native-sealed-tokenizer"

        @property
        def eot_token_id(self):
            return self.native.eot_token_id

        def loglikelihood(self, requests, **kwargs):
            return self.native.loglikelihood(requests, **kwargs)

        def loglikelihood_rolling(self, requests, **kwargs):
            return self.native.loglikelihood_rolling(requests, **kwargs)

        def generate_until(self, requests, **kwargs):
            return self.native.generate_until(requests, **kwargs)

        def apply_chat_template(self, *args, **kwargs):
            raise ValueError("The dense pretraining control is a base model, not a chat model")

    return MetisDenseHarnessLM()
