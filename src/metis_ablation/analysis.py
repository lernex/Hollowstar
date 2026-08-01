"""Routing analysis: the evidence that MoRE does what it claims.

Loss curves show MoRE is not worse.  This module produces the measurements that
show it is actually *adapting*, which is the part of the paper a referee will
push on hardest:

* depth and width distributions, and the joint histogram between them
* the depth-width correlation -- if the two axes move together perfectly, one
  is redundant and the three-dial framing is overstated, so this is reported
  whatever it says
* expert coalition transition matrices between consecutive passes, which is the
  only direct evidence for stage specialization (parse to manipulate to verify)
* halt calibration: predicted continuation probability against realized
  continuation
* per-pass active-token ratios

Everything is collected through forward hooks on the routing modules, so the
model itself carries no analysis branches in its hot path and any checkpoint can
be analyzed after the fact on a fixed held-out set.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch

from metis_training.model import AdaptiveDroplessMoE
from metis_training.model_config import Metis16Config


Tensor = torch.Tensor


@dataclass
class _PassRecord:
    """Top-1 expert per token for one (layer, pass), in the packed layout."""

    layer_index: int
    pass_index: int
    top_expert: Tensor
    chosen_k: Tensor


class RoutingAnalyzer:
    """Accumulates routing statistics across observed batches."""

    def __init__(self, config: Metis16Config, *, max_passes: int) -> None:
        self.config = config
        self.max_passes = int(max_passes)
        self.reset()

    def reset(self) -> None:
        experts = max(self.config.n_routed_experts, 1)
        widths = self.config.max_routed_k - self.config.min_routed_k + 1
        self.depth_histogram = torch.zeros(self.max_passes + 1, dtype=torch.float64)
        self.width_histogram = torch.zeros(
            self.max_passes + 1, widths, dtype=torch.float64
        )
        # depth x mean-width joint counts, for the correlation and the joint plot
        self.joint_depth_width = torch.zeros(
            self.max_passes + 1, widths, dtype=torch.float64
        )
        # One matrix per consecutive pass pair: depth 5 has four transitions.
        self.transitions = torch.zeros(
            max(self.max_passes - 1, 1), experts, experts, dtype=torch.float64
        )
        self.calibration_bins = 20
        self.calibration_predicted = torch.zeros(self.calibration_bins, dtype=torch.float64)
        self.calibration_realized = torch.zeros(self.calibration_bins, dtype=torch.float64)
        self.calibration_counts = torch.zeros(self.calibration_bins, dtype=torch.float64)
        self.active_ratio_sum = torch.zeros(self.max_passes, dtype=torch.float64)
        self.active_ratio_count = 0
        self._records: list[_PassRecord] = []
        self._pass_counter: dict[int, int] = {}
        self.observations = 0
        self._sum_depth = 0.0
        self._sum_width = 0.0
        self._sum_depth_sq = 0.0
        self._sum_width_sq = 0.0
        self._sum_depth_width = 0.0
        self._token_count = 0.0

    # ------------------------------------------------------------------ hooks

    @contextmanager
    def capture(self, model: Any) -> Iterator["RoutingAnalyzer"]:
        """Attach hooks to every routing module for the duration of one forward.

        Hooks are registered and removed around a single forward so the training
        loop pays nothing on the steps it does not analyze.
        """

        self._records.clear()
        self._pass_counter.clear()
        handles = []
        captured: list[AdaptiveDroplessMoE] = []
        for module in model.modules():
            if isinstance(module, AdaptiveDroplessMoE):
                module.capture_selection = True
                captured.append(module)
                handles.append(
                    module.register_forward_hook(self._make_hook(module), with_kwargs=True)
                )
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()
            for module in captured:
                module.capture_selection = False
                module._analysis_last_selection = None

    def _make_hook(self, module: AdaptiveDroplessMoE):
        def hook(mod, args, kwargs, output):  # noqa: ANN001 - torch hook signature
            del args, output
            layer_index = int(mod.layer_idx)
            # ``pass_index`` from the model is zero-based.
            pass_index = int(kwargs.get("pass_index", 0))
            state = getattr(mod, "_analysis_last_selection", None)
            if state is None:
                return
            top_expert, chosen_k = state
            self._records.append(
                _PassRecord(
                    layer_index=layer_index,
                    pass_index=pass_index,
                    top_expert=top_expert.detach().to("cpu"),
                    chosen_k=chosen_k.detach().to("cpu"),
                )
            )

        return hook

    # ------------------------------------------------------------- accumulate

    def observe(self, output: Any, attention_mask: Tensor) -> None:
        """Fold one analyzed forward into the running statistics."""

        self.observations += 1
        mask = attention_mask.detach().to("cpu")
        depths = output.chosen_depths.detach().to("cpu")
        valid = depths.masked_select(mask)
        if valid.numel():
            self.depth_histogram += torch.bincount(
                valid, minlength=self.max_passes + 1
            ).double()

        active = output.active_masks.detach().to("cpu")
        total_valid = float(mask.sum().item()) or 1.0
        for pass_index in range(min(active.shape[0], self.max_passes)):
            self.active_ratio_sum[pass_index] += (
                float(active[pass_index].sum().item()) / total_valid
            )
        self.active_ratio_count += 1

        widths = self.config.max_routed_k - self.config.min_routed_k + 1
        by_pass: dict[int, list[_PassRecord]] = {}
        for record in self._records:
            by_pass.setdefault(record.pass_index, []).append(record)

        # Width, keyed by the pass it was chosen in.
        for pass_index, records in by_pass.items():
            if pass_index >= self.max_passes:
                continue
            stacked = torch.cat([record.chosen_k.reshape(-1) for record in records])
            shifted = (stacked - self.config.min_routed_k).clamp(0, widths - 1)
            self.width_histogram[pass_index] += torch.bincount(
                shifted, minlength=widths
            ).double()

        # Depth against the token's own mean width: the correlation that decides
        # whether depth and width are genuinely distinct axes.
        first_pass = by_pass.get(0)
        if first_pass and valid.numel():
            per_token_width = torch.stack(
                [record.chosen_k.reshape(-1).double() for record in first_pass]
            ).mean(dim=0)
            flat_depth = depths.reshape(-1).double()
            flat_mask = mask.reshape(-1)
            if per_token_width.numel() == flat_depth.numel():
                d = flat_depth.masked_select(flat_mask)
                w = per_token_width.masked_select(flat_mask)
                self._sum_depth += float(d.sum())
                self._sum_width += float(w.sum())
                self._sum_depth_sq += float((d * d).sum())
                self._sum_width_sq += float((w * w).sum())
                self._sum_depth_width += float((d * w).sum())
                self._token_count += float(d.numel())
                bucket = (
                    (w.round().long() - self.config.min_routed_k)
                    .clamp(0, widths - 1)
                )
                for depth_value in range(1, self.max_passes + 1):
                    selector = d.long() == depth_value
                    if bool(selector.any()):
                        self.joint_depth_width[depth_value] += torch.bincount(
                            bucket.masked_select(selector), minlength=widths
                        ).double()

        # Expert coalition transitions between consecutive passes, aggregated
        # over layers.  A pass-invariant coalition produces a diagonal matrix;
        # stage specialization produces structured off-diagonal mass.
        #
        # Later passes run over a *packed* subset of tokens, so a record for
        # pass r+1 is a subsequence of pass r's rows, not a prefix.  Comparing
        # them positionally would pair unrelated tokens and manufacture
        # plausible-looking off-diagonal mass out of nothing.  ``active_masks``
        # carries the absolute layout for every pass, and the packer emits rows
        # in ascending flat-token order, so scattering each record back through
        # its own mask is an exact inverse.
        experts = max(self.config.n_routed_experts, 1)
        token_total = int(mask.numel())
        absolute_index: dict[int, Tensor] = {}
        for pass_index in range(min(active.shape[0], self.max_passes)):
            absolute_index[pass_index] = (
                active[pass_index].reshape(-1).nonzero(as_tuple=False).flatten()
            )

        def unpack(record: _PassRecord) -> Tensor | None:
            """Place a packed per-token record into the absolute token layout."""

            index = absolute_index.get(record.pass_index)
            values = record.top_expert.reshape(-1)
            if index is None:
                return None
            if values.numel() == token_total:
                # This pass was not packed; the record is already absolute.
                return values
            if values.numel() != index.numel():
                return None
            full = torch.full((token_total,), -1, dtype=values.dtype)
            full[index] = values
            return full

        for pass_index in range(self.max_passes - 1):
            current = by_pass.get(pass_index)
            following = by_pass.get(pass_index + 1)
            if not current or not following:
                continue
            after_by_layer = {record.layer_index: record for record in following}
            for before in current:
                after = after_by_layer.get(before.layer_index)
                if after is None:
                    continue
                source = unpack(before)
                target = unpack(after)
                if source is None or target is None:
                    continue
                # Only tokens that survived into the later pass have a
                # transition; a halted token has no successor coalition.
                both = (source >= 0) & (target >= 0)
                if not bool(both.any()):
                    continue
                flat = source[both] * experts + target[both]
                counts = torch.bincount(flat, minlength=experts * experts).double()
                self.transitions[pass_index] += counts.reshape(experts, experts)

    def observe_calibration(
        self,
        *,
        predicted: Tensor,
        continued: Tensor,
        valid: Tensor,
    ) -> None:
        """Fold a (probability, outcome) pair into the reliability diagram."""

        p = predicted.detach().to("cpu").reshape(-1).double()
        c = continued.detach().to("cpu").reshape(-1).double()
        v = valid.detach().to("cpu").reshape(-1).bool()
        p, c = p.masked_select(v), c.masked_select(v)
        if not p.numel():
            return
        bucket = (p * self.calibration_bins).long().clamp(0, self.calibration_bins - 1)
        self.calibration_counts += torch.bincount(
            bucket, minlength=self.calibration_bins
        ).double()
        self.calibration_predicted += torch.bincount(
            bucket, weights=p, minlength=self.calibration_bins
        ).double()
        self.calibration_realized += torch.bincount(
            bucket, weights=c, minlength=self.calibration_bins
        ).double()

    # ---------------------------------------------------------------- reports

    def depth_width_correlation(self) -> float | None:
        """Pearson r between per-token depth and per-token mean width."""

        n = self._token_count
        if n < 2:
            return None
        depth_var = self._sum_depth_sq - self._sum_depth * self._sum_depth / n
        width_var = self._sum_width_sq - self._sum_width * self._sum_width / n
        if depth_var <= 0 or width_var <= 0:
            return None
        covariance = self._sum_depth_width - self._sum_depth * self._sum_width / n
        return covariance / ((depth_var ** 0.5) * (width_var ** 0.5))

    def transition_off_diagonal_mass(self) -> list[float]:
        """Fraction of coalition transitions that actually change expert.

        Zero means the pathway axis is doing nothing: the same expert wins at
        every pass, and row 6 of the ladder should show no gap against row 5.
        """

        out: list[float] = []
        for pass_index in range(self.transitions.shape[0]):
            matrix = self.transitions[pass_index]
            total = float(matrix.sum())
            if total <= 0:
                out.append(0.0)
                continue
            out.append(1.0 - float(matrix.diagonal().sum()) / total)
        return out

    def calibration_curve(self) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for index in range(self.calibration_bins):
            count = float(self.calibration_counts[index])
            if count <= 0:
                continue
            rows.append(
                {
                    "bin": index / self.calibration_bins,
                    "count": count,
                    "mean_predicted": float(self.calibration_predicted[index]) / count,
                    "mean_realized": float(self.calibration_realized[index]) / count,
                }
            )
        return rows

    def report(self) -> dict[str, Any]:
        depth_total = float(self.depth_histogram.sum()) or 1.0
        return {
            "observations": self.observations,
            "depth_distribution": (self.depth_histogram / depth_total).tolist(),
            "mean_depth": float(
                (
                    self.depth_histogram
                    * torch.arange(self.max_passes + 1, dtype=torch.float64)
                ).sum()
                / depth_total
            ),
            "width_histogram_by_pass": self.width_histogram.tolist(),
            "joint_depth_width": self.joint_depth_width.tolist(),
            "depth_width_correlation": self.depth_width_correlation(),
            "transition_off_diagonal_mass": self.transition_off_diagonal_mass(),
            "active_token_ratio_by_pass": (
                self.active_ratio_sum / max(self.active_ratio_count, 1)
            ).tolist(),
            "halt_calibration": self.calibration_curve(),
        }

    def flush(self, path: Path, *, step: int) -> Path:
        payload = self.report()
        payload["step"] = step
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.reset()
        return path


__all__ = ["RoutingAnalyzer"]
