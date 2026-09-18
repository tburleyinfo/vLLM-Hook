# ==============================================================================
# VLLM HOOK SETUP AND DEPENDENCY MANAGER
# ==============================================================================
#
# PURPOSE:
# This cell prepares the environment to run vLLM and its custom plugins.
# It handles complex dependency conflicts common in Colab (e.g., pre-installed
# torch versions) and ensures the plugin code is loaded correctly.
#
# KEY ACTIONS:
# 1. Clones the vLLM-Hook repository if not present.
# 2. Installs a compatible version of vLLM and PyTorch for your GPU.
# 3. Cleans up old binary artifacts to prevent version conflicts.
# 4. Loads the plugin source code.
# 5. TRIGGERS A COLAB RESTART: The cell will restart the runtime to ensure
#    the new libraries are fully loaded into the kernel memory.
#
# EXPECTED BEHAVIOR:
# - The cell will run and install packages.
# - It may print "Restarting Colab runtime...".
# - The cell will stop abruptly, and the runtime will restart.
# - The restart is automatic; the code will resume from the top.
# ==============================================================================

from pathlib import Path
import importlib
import importlib.util
import os
import re
import shutil
import site
import subprocess
import sys
import time

# Configuration
REPO_URL = os.environ.get("VLLM_HOOK_REPO_URL", "https://github.com/IBM/vLLM-Hook.git")
REPO_BRANCH = os.environ.get("VLLM_HOOK_REPO_BRANCH", "main")
REPO_NAME = "vLLM-Hook"

# Environment Variables (Optional overrides)
COLAB_INSTALL_VLLM = os.environ.get("COLAB_INSTALL_VLLM", "")
VLLM_SPEC = os.environ.get("VLLM_SPEC", "vllm>=0.11,<0.19")
VLLM_TORCH_BACKEND = os.environ.get("VLLM_TORCH_BACKEND", "cu128")

IN_COLAB = "google.colab" in sys.modules
NOTEBOOK_DIR = Path.cwd()

# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------

def run(cmd, cwd=None, env=None):
    """
    Executes a shell command, printing output in real-time.
    Raises an error if the command fails.
    """
    cmd = [str(part) for part in cmd]
    print(f"> Running: {' '.join(cmd)}", flush=True)

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    tail = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
        tail = tail[-120:]

    returncode = process.wait()
    if returncode:
        tail_text = "\n".join(tail)
        raise RuntimeError(
            f"Command failed with exit code {returncode}: {' '.join(cmd)}\n\n"
            f"Last output lines:\n{tail_text}"
        )

def run_capture(cmd):
    """Executes a command and returns stdout/stderr without printing."""
    return subprocess.run([str(part) for part in cmd], text=True, capture_output=True, check=False)

def norm(name):
    """Normalize package names for comparison."""
    return name.lower().replace("_", "-")

def package_from_req_line(line: str) -> str:
    """Extract package name from a requirement string (e.g., 'torch>=1.0' -> 'torch')."""
    stripped = line.strip()
    # Split on version specifiers
    package = re.split(r"==|>=|<=|~=|!=|<|>|\[", stripped, maxsplit=1)[0]
    return norm(package.strip())

def _repo_remote_matches(repo_root: Path, expected_remote: str) -> bool:
    """Checks if the git repo's origin matches the expected URL."""
    try:
        url = subprocess.run(
            ["git", "-C", str(repo_root), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().removesuffix(".git")
    except Exception:
        return False
    return url == expected_remote

def _find_existing_repo_root(start_dir: Path, expected_remote: str):
    """Searches up the directory tree for a matching git repo."""
    for candidate in [start_dir, *start_dir.parents]:
        if (candidate / ".git").exists() and _repo_remote_matches(candidate, expected_remote):
            return candidate
    return None

def assert_cuda_runtime():
    """Ensures a GPU is available before proceeding."""
    try:
        import torch
    except Exception:
        torch = None

    has_cuda = bool(torch is not None and torch.cuda.is_available())
    has_cudart = importlib.util.find_spec("nvidia.cuda_runtime") is not None

    if not has_cuda and not has_cudart:
        raise RuntimeError(
            "No CUDA GPU detected. "
            "In Colab, go to Runtime > Change runtime type and select T4 GPU (or better), "
            "then re-run the entire notebook from the beginning."
        )

# --------------------------------------------------------------------------
# 1. Repository Setup
# --------------------------------------------------------------------------

expected_remote = REPO_URL.removesuffix(".git")
existing_repo_root = _find_existing_repo_root(NOTEBOOK_DIR, expected_remote)

if IN_COLAB:
    if existing_repo_root is not None:
        REPO_ROOT = existing_repo_root
        print(f"✓ Reusing existing repo at: {REPO_ROOT}")
    else:
        REPO_ROOT = Path("/content") / REPO_NAME
        if not REPO_ROOT.exists():
            print(f"Cloning {REPO_URL} (branch: {REPO_BRANCH})...")
            run(["git", "clone", "--branch", REPO_BRANCH, REPO_URL, str(REPO_ROOT)])
        elif not _repo_remote_matches(REPO_ROOT, expected_remote):
            print(f"Remote mismatch detected. Replacing clone...")
            shutil.rmtree(REPO_ROOT)
            run(["git", "clone", "--branch", REPO_BRANCH, REPO_URL, str(REPO_ROOT)])
        else:
            print(f"Refreshing existing clone...")
            run(["git", "-C", str(REPO_ROOT), "fetch", "origin", REPO_BRANCH])
            run(["git", "-C", str(REPO_ROOT), "checkout", REPO_BRANCH])
            run(["git", "-C", str(REPO_ROOT), "pull", "--ff-only", "origin", REPO_BRANCH])

    NOTEBOOK_DIR = REPO_ROOT / "notebooks"
    os.chdir(NOTEBOOK_DIR)
    print(f"Working directory set to: {NOTEBOOK_DIR}")
else:
    REPO_ROOT = NOTEBOOK_DIR.parent

PKG_DIR = REPO_ROOT / "vllm_hook_plugins"
REQ_FILE = REPO_ROOT / "requirement.txt"
FILTERED_REQ_FILE = Path("/tmp/vllm_hook_colab_requirements.txt")
COLAB_RESTART_MARKER = Path("/tmp/vllm_hook_colab_binary_deps_restarted")

print(f"\n--- Environment Summary ---")
print(f"Running in Colab: {IN_COLAB}")
print(f"Repo Root: {REPO_ROOT}")
print(f"Plugin Dir: {PKG_DIR}")

if IN_COLAB:
    assert_cuda_runtime()

# --------------------------------------------------------------------------
# 2. Plugin Directory Validation
# --------------------------------------------------------------------------

if not PKG_DIR.exists():
    raise FileNotFoundError(
        f"Plugin directory not found at {PKG_DIR}. "
        "Please ensure the repository was cloned correctly."
    )

if shutil.which("git") is None and IN_COLAB:
    raise RuntimeError("git is required but unavailable in this runtime.")

# --------------------------------------------------------------------------
# 3. Dependency Management
# --------------------------------------------------------------------------

if REQ_FILE.exists():
    keep = []
    # These packages are managed by Colab's pre-installed environment.
    # Installing them again can cause conflicts.
    blocked = {"vllm", "torch", "torchvision", "torchaudio", "numpy", "scipy", "protobuf"}

    for line in REQ_FILE.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            keep.append(line)
            continue

        package = package_from_req_line(stripped)
        if package in blocked:
            print(f"⏭ Skipping managed dependency: {line}")
            continue
        keep.append(line)

    FILTERED_REQ_FILE.write_text("\n".join(keep) + "\n")
    print(f"Installing filtered requirements from {REQ_FILE.name}...")
    run([sys.executable, "-m", "pip", "install", "-r", str(FILTERED_REQ_FILE)])
else:
    print("⚠ Warning: No requirement.txt found; skipping custom dependency installation.")

# Ensure protobuf is at the correct version
print("Ensuring protobuf version compatibility...")
run([sys.executable, "-m", "pip", "install", "--force-reinstall", "protobuf>=5.29.6,<6.30"])

# --------------------------------------------------------------------------
# 4. vLLM & PyTorch Installation
# --------------------------------------------------------------------------

if COLAB_INSTALL_VLLM:
    print(f"Installing user-specified vLLM: {COLAB_INSTALL_VLLM}")
    run([sys.executable, "-m", "pip", "install", COLAB_INSTALL_VLLM])
else:
    print(f"\nInstalling vLLM and PyTorch for {VLLM_TORCH_BACKEND}...")

    # 1. Uninstall existing conflicting versions
    run([sys.executable, "-m", "pip", "uninstall", "-y", "vllm", "torch", "torchvision", "torchaudio"])

    # 2. Manually remove leftover binary artifacts that pip might miss
    for site_dir in site.getsitepackages():
        site_path = Path(site_dir)
        leftovers = [
            site_path / "vllm", *site_path.glob("vllm-*.dist-info"),
            site_path / "torch", *site_path.glob("torch-*.dist-info"),
            site_path / "torchvision", *site_path.glob("torchvision-*.dist-info"),
            site_path / "torchaudio", *site_path.glob("torchaudio-*.dist-info"),
        ]
        for leftover in leftovers:
            if leftover.exists():
                print(f"  Cleaning up leftover: {leftover.name}")
                if leftover.is_dir():
                    shutil.rmtree(leftover)
                else:
                    leftover.unlink()

    # 3. Force clear GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    # 4. Install using 'uv' (faster than pip) with specific CUDA backend
    print("  Downloading and installing packages with 'uv'...")
    run([sys.executable, "-m", "pip", "install", "-U", "uv"])
    run([
        "uv", "pip", "install",
        "--system",
        VLLM_SPEC,
        "torch", "torchvision", "torchaudio",
        f"--torch-backend={VLLM_TORCH_BACKEND}",
    ])

    # 5. Verify Scipy/Numpy
    scipy_check = run_capture([
        sys.executable, "-c", "import numpy, scipy; print('numpy', numpy.__version__); print('scipy', scipy.__version__)"
    ])
    if scipy_check.returncode:
        print("  Detected numpy/scipy issues; upgrading...")
        run([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "numpy", "scipy"])

# Verify Installation
print("\nVerifying installation...")
run([
    sys.executable,
    "-c",
    "import torch, vllm; print('✓ torch:', torch.__version__); print('✓ vllm:', getattr(vllm, '__version__', 'unknown'))"
])

# --------------------------------------------------------------------------
# 5. Plugin Loading
# --------------------------------------------------------------------------

# Strategy: Add path to sys.path to allow immediate import,
# then ensure metadata is installed for subprocesses later.
plugin_src_dir = str(PKG_DIR.resolve())
if plugin_src_dir not in sys.path:
    sys.path.insert(0, plugin_src_dir)
importlib.invalidate_caches()

try:
    spec = importlib.util.spec_from_file_location("vllm_hook_plugins", PKG_DIR / "__init__.py")
    if spec:
        importlib.util.module_from_spec(spec)
        print("✓ Plugin module loaded successfully from source path.")
except Exception as e:
    print(f"⚠ Warning: Initial import check failed ({e}). This is expected if compilation is needed later.")

# Ensure the package is registered in the environment (metadata)
# This is necessary for tools that rely on `importlib.metadata`.
print("Registering plugin package in environment...")
run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(PKG_DIR)])

print(f"Plugin Source: {plugin_src_dir}")
print(f"Python Exec  : {sys.executable}")

# --------------------------------------------------------------------------
# 6. Runtime Restart (Critical for Colab)
# --------------------------------------------------------------------------
# We restart the runtime to ensure the newly installed binary libraries
# are loaded into a fresh Python interpreter. This prevents "dirty state" issues.
if IN_COLAB and not COLAB_RESTART_MARKER.exists():
    COLAB_RESTART_MARKER.write_text("1\n")
    print("\n" + "="*50)
    print("RESTARTING COLAB RUNTIME...")
    print("This ensures the new vLLM/Torch binaries are fully loaded.")
    print("Do not interrupt this process.")
    print("="*50)

    time.sleep(1) # Allow output buffer to flush
    sys.exit(0) # Terminate this process to trigger Colab restart
