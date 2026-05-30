"""Check that the local ARTLib environment is working."""

from __future__ import annotations

import importlib.metadata
import sys


def get_artlib_version() -> str:
    try:
        import artlib

        version = getattr(artlib, "__version__", None)
        if version:
            return str(version)
    except Exception:
        pass

    try:
        return importlib.metadata.version("artlib")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> int:
    print(f"Python version: {sys.version.split()[0]}")

    try:
        from artlib import FuzzyART
    except Exception as exc:
        print(f"ARTLib import failed: {exc}")
        return 1

    print(f"ARTLib import: {FuzzyART.__module__}.{FuzzyART.__name__}")
    print(f"ARTLib version: {get_artlib_version()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())