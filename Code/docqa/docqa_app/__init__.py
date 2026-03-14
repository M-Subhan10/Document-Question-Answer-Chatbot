from __future__ import annotations


def build_app(*args, **kwargs):
    from .ui import build_app as _build_app

    return _build_app(*args, **kwargs)


__all__ = ["build_app"]
