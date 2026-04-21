"""DeepSpeed-MII serve backend (v0.27.0).

MII (Model Implementations for Inference) provides high-throughput serving
with tensor parallelism. See https://github.com/deepspeedai/DeepSpeed-MII.

All imports are lazy so that `soup --help` stays fast.
"""

from __future__ import annotations

import sys
from typing import Any


def is_mii_available() -> bool:
    """Return True when the ``mii`` package is importable.

    Honours test-injected stubs: if ``sys.modules["mii"]`` is explicitly set
    to ``None`` (pytest's idiom for "pretend this module is absent"), the
    ``import mii`` statement below will raise ``ImportError`` without
    consulting disk.
    """
    if "mii" in sys.modules and sys.modules["mii"] is None:
        return False
    try:
        import mii  # noqa: F401
    except ImportError:
        return False
    return True


def create_mii_pipeline(
    model_path: str,
    tensor_parallel: int = 1,
    max_length: int = 4096,
    replica_num: int = 1,
) -> Any:
    """Create a DeepSpeed-MII pipeline.

    Args:
        model_path: HF model id or local path.
        tensor_parallel: TP size (must evenly divide GPU count).
        max_length: Max sequence length the pipeline will handle.
        replica_num: Number of replicas (for multi-node).

    Raises:
        ImportError: If the ``deepspeed-mii`` package is not installed.
    """
    if not is_mii_available():
        raise ImportError(
            "deepspeed-mii is not installed. "
            "Install with: pip install 'soup-cli[mii]' "
            "or pip install deepspeed-mii"
        )

    import mii

    return mii.pipeline(
        model_path,
        tensor_parallel=tensor_parallel,
        max_length=max_length,
        replica_num=replica_num,
    )
