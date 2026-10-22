"""Select the MIA plugin implementation without importing backend-incompatible internals."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def register() -> None:
    """Register MIA's PyTorch-runner patches only when the Metal plugin is not selected."""
    plugins = {
        name.strip().lower()
        for name in os.environ.get("VLLM_PLUGINS", "").split(",")
    }
    if "metal" in plugins:
        logger.info(
            "MIA's PyTorch runner instrumentation is disabled with vLLM-Metal; "
            "generation remains available through the Metal platform plugin."
        )
        return

    from mia.core._plugin import register as register_torch_runner_plugin

    register_torch_runner_plugin()
