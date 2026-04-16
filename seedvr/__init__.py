from __future__ import annotations

__all__ = ["SeedVRPipeline"]


def __getattr__(name: str):
    if name == "SeedVRPipeline":
        from .pipeline import SeedVRPipeline

        return SeedVRPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
