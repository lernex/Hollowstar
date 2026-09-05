"""Load the new evaluator without replacing the checkpoint's native packages."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--native-source", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    native = args.native_source.resolve(strict=True) / "src"
    evaluator = Path(__file__).resolve().parents[1] / "src" / "metis_ablation"
    sys.path.insert(0, str(native))
    import metis_ablation

    if Path(metis_ablation.__file__).resolve().parent != native / "metis_ablation":
        raise RuntimeError("The original ablation package was not imported")
    # Only the two new evaluation modules come from the evaluator commit.
    metis_ablation.__path__.append(str(evaluator))
    from metis_ablation.evaluate import main as evaluate

    return evaluate(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
