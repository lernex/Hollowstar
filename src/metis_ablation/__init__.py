"""MoRE paper ablation campaign.

This package trains the axis-isolating ladder described in
``docs/papers/more/ablation_campaign.md``.  It deliberately does not reuse
``metis_training.train``: the production trainer is bound to the immutable 1T
release contract, its phase boundaries, and its autotune lineage, and the 1T
Praxis/Logos runs must not be destabilized by research plumbing.  What is shared
is everything that matters for comparability -- the model, the release stream,
the optimizer, the precision policy, and the FLOP accounting.

Wave 1 runs after Praxis and Logos release the full 512-APU Portage machine.
"""

from .specs import (
    ABLATION_LADDER,
    AblationSpec,
    GLOBAL_BATCH_SEQUENCES,
    proxy_config,
    spec_by_name,
    validate_allocation,
)

__all__ = [
    "ABLATION_LADDER",
    "AblationSpec",
    "GLOBAL_BATCH_SEQUENCES",
    "proxy_config",
    "spec_by_name",
    "validate_allocation",
]
