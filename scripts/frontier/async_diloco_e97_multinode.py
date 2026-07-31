#!/usr/bin/env python3
"""Stable async DiLoCo E97 multi-node entrypoint for Frontier wrappers.

This production-bound entrypoint must not delegate to the synthetic debug
harness.  Keep the wrapper thin so Frontier launch scripts can call the real
trainer while tests can assert this file does not silently run protocol-only
updates.
"""

from __future__ import annotations

from e97_async_diloco_train import main


if __name__ == "__main__":
    raise SystemExit(main())
