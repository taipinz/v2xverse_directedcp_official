#!/usr/bin/env python3
"""Validate the runtime environment required by V2Xverse closed-loop evaluation."""

from __future__ import annotations

import argparse
import glob
import importlib.util
import platform
import sys
from pathlib import Path


REQUIRED_MODULES = (
    "torch",
    "yaml",
    "pygame",
    "py_trees",
    "dictor",
    "ephem",
    "cv2",
    "networkx",
    "PIL",
)


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_carla_egg(dist_dir: Path) -> tuple[str | None, list[str]]:
    py_tag = f"py{sys.version_info.major}.{sys.version_info.minor}"
    matches = sorted(glob.glob(str(dist_dir / f"carla-*-{py_tag}-linux-x86_64.egg")))
    available = sorted(glob.glob(str(dist_dir / "carla-*.egg")))
    return (matches[0] if matches else None), available


def module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def validate(verbose: bool = True) -> int:
    repo_root = find_repo_root()
    carla_root = repo_root / "external_paths" / "carla_root"
    data_root = repo_root / "external_paths" / "data_root"
    dist_dir = carla_root / "PythonAPI" / "carla" / "dist"
    current_python = Path(sys.executable).resolve()

    issues: list[str] = []
    notes: list[str] = []

    carla_egg, available_eggs = find_carla_egg(dist_dir)
    if not carla_root.exists():
        issues.append(f"missing CARLA path: {carla_root}")
    if not data_root.exists():
        notes.append(f"data symlink/path is missing: {data_root}")
    if not dist_dir.exists():
        issues.append(f"missing CARLA Python dist directory: {dist_dir}")
    elif carla_egg is None:
        py_tag = f"py{sys.version_info.major}.{sys.version_info.minor}"
        issues.append(
            "no CARLA egg matches the current interpreter "
            f"({py_tag}, python {platform.python_version()})"
        )
        if available_eggs:
            notes.append("available CARLA eggs:\n  - " + "\n  - ".join(available_eggs))
            if any("py3.7" in egg for egg in available_eggs):
                notes.append(
                    "the bundled CARLA 0.9.10 egg targets Python 3.7; "
                    "using it from Python 3.8+ can segfault"
                )
    else:
        sys.path.insert(0, carla_egg)
        try:
            import carla  # type: ignore

            if not hasattr(carla, "Client"):
                issues.append(f"CARLA module imported from {carla_egg} but lacks Client")
        except Exception as exc:  # pragma: no cover
            issues.append(f"failed to import CARLA from {carla_egg}: {exc}")

    missing_modules = [name for name in REQUIRED_MODULES if not module_exists(name)]
    if missing_modules:
        issues.append("missing Python modules: " + ", ".join(missing_modules))

    if verbose:
        print(f"[closed-loop-check] repo: {repo_root}")
        print(f"[closed-loop-check] python: {current_python} ({platform.python_version()})")
        print(f"[closed-loop-check] carla_root: {carla_root}")
        if carla_egg:
            print(f"[closed-loop-check] matched_carla_egg: {carla_egg}")
        if issues:
            print("[closed-loop-check] status: FAIL")
            for item in issues:
                print(f"  - {item}")
            for item in notes:
                print(f"  - note: {item}")
            print(
                "[closed-loop-check] hint: export V2XVERSE_PYTHON="
                "$HOME/.local/share/mamba/envs/v2xverse/bin/python "
                "or activate a Python 3.7 environment before running closed-loop evaluation"
            )
        else:
            print("[closed-loop-check] status: OK")
            for item in notes:
                print(f"  - note: {item}")

    return 1 if issues else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true", help="suppress detailed output")
    args = parser.parse_args()
    return validate(verbose=not args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
