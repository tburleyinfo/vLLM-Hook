# Legacy CUDA installation tests

The adjacent installation scripts and `cross-platform-install/gpu_ci.py` target the former CPU
and CUDA profiles. The vLLM-Metal port removes those profiles, and the CUDA CI workflow has been
removed. These files are retained as historical reference only and are not runnable using the
current project metadata.

There is no supplied Apple Silicon CI runner, Metal test recipe, or Metal instrumentation test
suite. For supported Metal generation usage and the current unsupported MIA capabilities, see the
root [`README.md`](../README.md).
