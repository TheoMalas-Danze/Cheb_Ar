"""Helpers shared by the per-solver pipelines."""


def error_entry(exc, **coords):
    """Uniform placeholder for a point that failed: its coordinates + ``error``."""
    entry = {k: v for k, v in coords.items()}
    entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry
