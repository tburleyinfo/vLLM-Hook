Local mlx-lm Override for vllm-metal
====================================

Goal
----
Point vllm-metal's active virtual environment to your local mlx-lm checkout
instead of the published package.

Dependency Source (Validation Reference)
----------------------------------------
vllm-metal declares mlx-lm in:
- /Users/timothyburley/opensource/vllm-metal/pyproject.toml:32

The dependency block begins at:
- /Users/timothyburley/opensource/vllm-metal/pyproject.toml:29

Instructions
------------
# 1) Activate the same environment used to run vllm-metal:
   source /Users/timothyburley/opensource/vllm-metal/.venv-vllm-metal/bin/activate

# 2) Remove any currently installed mlx-lm package entry:
   pip uninstall -y mlx-lm mlx_lm

# 3) Install your local mlx-lm checkout in editable mode:
   pip install -e /Users/timothyburley/opensource/mlx-lm

# 4) Verify venv wiring:
   pip list --editable
   pip show mlx-lm

Expected verification:
- "pip list --editable" includes mlx-lm with project location:
  /Users/timothyburley/opensource/mlx-lm
- "pip show mlx-lm" reports it as installed in the current venv.

Important Notes
---------------
- Reinstalling vllm-metal with dependencies can replace your editable mlx-lm.
  During local development, prefer:
  pip install -e /Users/timothyburley/opensource/vllm-metal --no-deps

- If you need to revert to the published mlx-lm package:
  pip uninstall -y mlx-lm mlx_lm
  pip install -U "mlx-lm>=0.20.0"

Install/Reinstall Checkpoints During Test Cycle
-----------------------------------------------
Use these checkpoints whenever behavior looks stale or imports are inconsistent.

Checkpoint A: Ensure vllm-metal venv is active
----------------------------------------------
```bash
source /Users/timothyburley/opensource/vllm-metal/.venv-vllm-metal/bin/activate
python -c "import sys; print(sys.executable)"
```

Checkpoint B: Re-point mlx-lm to local checkout
------------------------------------------------
```bash
python -m pip uninstall -y mlx-lm mlx_lm
python -m pip install -e /Users/timothyburley/opensource/mlx-lm
python -m pip list --editable
python -m pip show mlx-lm
```

Checkpoint C: Reinstall/refresh vllm-metal from local source
-------------------------------------------------------------
Preferred (preserve local mlx-lm override):
```bash
python -m pip install -e /Users/timothyburley/opensource/vllm-metal --no-deps
```

If editable build fails due to missing build backend (for example maturin),
run tests with source override instead:
```bash
PYTHONPATH=/Users/timothyburley/opensource/vllm-metal:$PYTHONPATH \
python -m pytest metal_tests/test_local_mlx_lm_wiring_metal.py -vv
```

Checkpoint D: Verify which vllm-metal code is imported
------------------------------------------------------
```bash
python - <<'PY'
import inspect
import vllm_metal
print("vllm_metal:", vllm_metal.__file__)
try:
    import vllm_metal.v1.model_runner as mr
    print("model_runner:", mr.__file__)
except Exception as e:
    print("model_runner import error:", type(e).__name__, e)
PY
```

Checkpoint E: Return to packaged state
--------------------------------------
```bash
python -m pip uninstall -y mlx-lm mlx_lm vllm-metal
python -m pip install -U "mlx-lm>=0.20.0" "vllm-metal"
python -m pip list --editable
```
