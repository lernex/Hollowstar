from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .util import canonical_json_bytes


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _expand_string(value: str, environment: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        fallback = match.group(2)
        resolved = environment.get(name)
        if resolved:
            return resolved
        if fallback is not None:
            return fallback
        raise RuntimeError(f"Required environment variable {name} is not set")

    return _ENV.sub(replace, value)


def _expand(value: Any, environment: Mapping[str, str]) -> Any:
    if isinstance(value, dict):
        return {str(key): _expand(item, environment) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, environment) for item in value]
    if isinstance(value, str):
        return _expand_string(value, environment)
    return value


@dataclass(frozen=True)
class FamilyTopology:
    name: str
    nodes: int
    world_size: int
    relative_node: int
    expert_parallel_size: int
    expert_replicas: int
    manifest: Path


@dataclass(frozen=True)
class PortageConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def schema(self) -> str:
        return str(self.raw["schema"])

    @property
    def partition(self) -> str:
        return str(self.raw["site"]["partition"])

    @property
    def accelerators_per_node(self) -> int:
        return int(self.raw["site"]["accelerators_per_node"])

    @property
    def repository(self) -> Path:
        return repository_root()

    @property
    def lustre_root(self) -> Path:
        return Path(self.raw["storage"]["lustre_root"]).expanduser().resolve()

    @property
    def release_root(self) -> Path:
        return Path(self.raw["storage"]["release_root"]).expanduser().resolve()

    @property
    def state_root(self) -> Path:
        return Path(self.raw["storage"]["state_root"]).expanduser().resolve()

    @property
    def posttraining_release_index(self) -> Path:
        configured = str(
            self.raw["storage"]["posttraining_release_index"]
        ).strip()
        if configured:
            return Path(configured).expanduser().resolve()
        return (
            self.lustre_root
            / "releases"
            / "metis-1.6-posttraining-r1"
            / "RELEASE_INDEX.json"
        ).resolve()

    @property
    def training_contract(self) -> Path:
        return (self.repository / self.raw["training"]["pretraining_contract"]).resolve()

    @property
    def runtime_policy(self) -> Path:
        return (self.repository / self.raw["training"]["runtime_policy"]).resolve()

    @property
    def posttraining_contract(self) -> Path:
        return (
            self.repository / self.raw["training"]["posttraining_contract"]
        ).resolve()

    @property
    def trainer_argv(self) -> tuple[str, ...]:
        override = os.environ.get("METIS_TRAIN_COMMAND", "").strip()
        if override:
            import shlex

            return tuple(shlex.split(override))
        return tuple(str(item) for item in self.raw["training"]["command"])

    @property
    def families(self) -> tuple[FamilyTopology, ...]:
        rows: list[FamilyTopology] = []
        for name in ("praxis", "logos"):
            item = self.raw["families"][name]
            rows.append(
                FamilyTopology(
                    name=name,
                    nodes=int(item["nodes"]),
                    world_size=int(item["world_size"]),
                    relative_node=int(item["relative_node"]),
                    expert_parallel_size=int(item["expert_parallel_size"]),
                    expert_replicas=int(item["expert_replicas"]),
                    manifest=(self.repository / item["manifest"]).resolve(),
                )
            )
        return tuple(rows)

    def validate(self) -> None:
        if self.schema != "metis.portage-training/v1":
            raise RuntimeError(f"Unexpected Portage config schema: {self.schema}")
        if self.partition != "parry":
            raise RuntimeError("The locked Portage training partition must be parry")
        if self.accelerators_per_node != 4:
            raise RuntimeError("Metis-1.6 topology is locked to four MI300A APUs per node")
        if int(self.raw["site"]["cpu_cores_per_node"]) != 24:
            raise RuntimeError("Metis-1.6 Portage contract expects 24 Zen 4 cores per node")
        families = {family.name: family for family in self.families}
        praxis = families["praxis"]
        logos = families["logos"]
        if (
            (praxis.nodes, praxis.world_size, praxis.relative_node)
            != (32, 128, 0)
            or (praxis.expert_parallel_size, praxis.expert_replicas) != (128, 1)
            or (logos.nodes, logos.world_size, logos.relative_node)
            != (96, 384, 32)
            or (logos.expert_parallel_size, logos.expert_replicas) != (192, 2)
        ):
            raise RuntimeError("Praxis-128 / Logos-384 topology differs from the locked plan")
        if sum(item.nodes for item in self.families) != 128:
            raise RuntimeError("Family topology does not consume exactly 128 Portage nodes")
        if sum(item.world_size for item in self.families) != 512:
            raise RuntimeError("Family topology does not consume exactly 512 MI300A APUs")
        if not self.training_contract.is_file():
            raise RuntimeError(f"Missing pretraining contract: {self.training_contract}")
        if not self.runtime_policy.is_file():
            raise RuntimeError(f"Missing Portage runtime policy: {self.runtime_policy}")
        if not self.posttraining_contract.is_file():
            raise RuntimeError(
                f"Missing post-training contract: {self.posttraining_contract}"
            )
        forbidden = {
            Path("/").resolve(),
            Path("/lus").resolve(),
            Path("/lus/lustre1").resolve(),
        }
        if self.lustre_root in forbidden:
            raise RuntimeError(f"Unsafe Lustre root: {self.lustre_root}")
        for label, path in (
            ("release_root", self.release_root),
            ("posttraining_release_index", self.posttraining_release_index),
            ("state_root", self.state_root),
        ):
            try:
                path.relative_to(self.lustre_root)
            except ValueError as exc:
                raise RuntimeError(f"{label} must be beneath the explicit Lustre root") from exc
        if (
            self.release_root == self.lustre_root
            or self.posttraining_release_index == self.lustre_root
            or self.state_root == self.lustre_root
        ):
            raise RuntimeError("Release/state paths may not equal the Lustre root")
        if not self.trainer_argv:
            raise RuntimeError("Training command is empty")
        if int(self.raw["autonomy"]["maximum_automatic_restarts"]) < 0:
            raise RuntimeError("maximum_automatic_restarts may not be negative")
        if not 1 <= int(
            self.raw["autonomy"]["maximum_deferred_materialization_attempts"]
        ) <= 10:
            raise RuntimeError(
                "maximum_deferred_materialization_attempts must be in [1,10]"
            )
        if not 1 <= int(
            self.raw["autonomy"][
                "maximum_deferred_materializations_per_family"
            ]
        ) <= 128:
            raise RuntimeError(
                "maximum_deferred_materializations_per_family must be in [1,128]"
            )
        if int(self.raw["bringup"]["mhc_canary_tokens"]) <= 0:
            raise RuntimeError("bringup.mhc_canary_tokens must be positive")
        for field in (
            "mhc_max_forward_relative_error",
            "mhc_max_backward_relative_error",
        ):
            value = float(self.raw["bringup"][field])
            if not 0.0 < value < 1.0:
                raise RuntimeError(f"bringup.{field} must be in (0, 1)")
        if float(self.raw["bringup"]["mhc_minimum_speedup"]) < 1.0:
            raise RuntimeError(
                "bringup.mhc_minimum_speedup must require throughput-positive fusion"
            )
        canonical_json_bytes(self.raw)


def load_portage_config(
    path: str | Path | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> PortageConfig:
    source = (
        repository_root() / "configs" / "metis16" / "portage-training.yaml"
        if path is None
        else Path(path).expanduser().resolve()
    )
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Portage config must be a mapping: {source}")
    resolved_environment = dict(
        os.environ if environment is None else environment
    )
    explicit_lustre_root = resolved_environment.get(
        "METIS_LUSTRE_ROOT", ""
    ).strip()
    if explicit_lustre_root:
        root = Path(explicit_lustre_root).expanduser()
        if not resolved_environment.get("METIS_DATA_RELEASE", "").strip():
            resolved_environment["METIS_DATA_RELEASE"] = str(
                root / "releases" / "metis-1.6-data-r1"
            )
        if not resolved_environment.get(
            "METIS_PORTAGE_STATE_ROOT", ""
        ).strip():
            resolved_environment["METIS_PORTAGE_STATE_ROOT"] = str(
                root / "training" / "metis-1.6" / "portage"
            )
    expanded = _expand(payload, resolved_environment)
    config = PortageConfig(path=source, raw=expanded)
    config.validate()
    return config
