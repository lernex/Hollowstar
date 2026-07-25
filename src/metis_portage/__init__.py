"""Autonomous, fail-closed Portage bring-up and launch control.

The package deliberately contains no model implementation.  It inventories the
live HPE/Slurm/ROCm site, proves the required communication and release
contracts, selects a measured trainer profile, and only then starts the two
Metis-1.6 family jobs.
"""

from .config import PortageConfig, load_portage_config

__all__ = ["PortageConfig", "load_portage_config"]
