"""End-to-end pipeline runner: dataset → baseline → improved → custom.

Runs every stage of the lab in sequence and stops with a non-zero exit code
the moment any step fails. Each stage is invoked as a subprocess so logs are
streamed live and per-script failures surface immediately.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
STEPS: list[list[str]] = [
    [sys.executable, "prepare_dataset.py"],
    [sys.executable, "train_baseline.py"],
    [sys.executable, "evaluate.py", "yolo11n_baseline", "rtdetr_l_baseline"],
    [sys.executable, "train_improved.py"],
    [sys.executable, "evaluate.py", "yolo11n_improved", "rtdetr_l_improved"],
    [sys.executable, "custom_detector.py"],
    [sys.executable, "evaluate.py"],
]


def main() -> int:
    """Run every pipeline step in order and short-circuit on first failure."""
    for step in STEPS:
        header = " ".join(step[1:])
        print(f"\n===== {header} =====\n", flush=True)
        result = subprocess.run(step, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(
                f"\nStep `{header}` failed with exit code {result.returncode}",
                file=sys.stderr,
            )
            return result.returncode
    print("\nAll steps finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
