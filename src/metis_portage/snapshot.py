from __future__ import annotations

import argparse

from .telemetry import write_node_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture one ROCm/CXI node snapshot.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    write_node_snapshot(args.output, label=args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
