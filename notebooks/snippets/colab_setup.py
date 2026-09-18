# ==============================================================================
# VLLM HOOK SETUP AND DEPENDENCY MANAGER
# ==============================================================================
import importlib
import importlib.metadata as importlib_metadata
import os
import re
import subprocess
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_URL = os.environ.get("VLLM_HOOK_REPO_URL", "https://github.com/IBM/vLLM-Hook.git")
REPO_BRANCH = os.environ.get("VLLM_HOOK_REPO_BRANCH", "main")
REPO_DIR = Path(os.environ.get("VLLM_HOOK_REPO_DIR", "/content/vLLM-Hook"))
PLUGIN_SRC = REPO_DIR / "vllm_hook_plugins"

# These must be set before vLLM is imported for the first time in this kernel.
os.environ["VLLM_USE_V1"] = "1"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("HF_HOME", "/content/.cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/content/.cache/huggingface/hub")


def run(cmd, cwd=None, check=True):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    result = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        print(result.stdout, end="")
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout)
    return result


def normalized_package_name(requirement_line):
    line = requirement_line.split("#", 1)[0].strip()
    if not line or line.startswith("-"):
        return ""
    return re.split(r"[<>=!~;\[]", line, maxsplit=1)[0].strip().lower().replace("_", "-")


def version_tuple(version):
    parts = []
    for part in re.split(r"[.+-]", version):
        if part.isdigit():
            parts.append(int(part))
        else:
            break
    return tuple(parts)


def protobuf_needs_pin():
    try:
        current = importlib_metadata.version("protobuf")
    except importlib_metadata.PackageNotFoundError:
        return True
    parsed = version_tuple(current)
    return not ((5, 29, 6) <= parsed < (6, 30))


def fail_if_already_imported(package_names):
    imported = sorted(
        name for name in package_names
        if name in sys.modules or any(mod.startswith(name + ".") for mod in sys.modules)
    )
    if imported:
        raise RuntimeError(
            "The setup cell needs to run before importing "
            + ", ".join(imported)
            + ". Restart the runtime once, then use Runtime > Run all."
        )


if IN_COLAB:
    fail_if_already_imported(["torch", "torchvision", "torchaudio", "vllm"])

    if not REPO_DIR.exists():
        run(["git", "clone", "--branch", REPO_BRANCH, REPO_URL, REPO_DIR])
    else:
        run(["git", "-C", REPO_DIR, "fetch", "origin", REPO_BRANCH])
        run(["git", "-C", REPO_DIR, "checkout", REPO_BRANCH])
        run(["git", "-C", REPO_DIR, "pull", "--ff-only", "origin", REPO_BRANCH])

    filtered_req = Path("/tmp/vllm_hook_colab_requirements.txt")
    req = REPO_DIR / "requirement.txt"
    if req.exists():
        keep = []
        skip_packages = {"vllm", "torch", "torchvision", "torchaudio", "protobuf", "pillow", "pil"}
        for line in req.read_text(encoding="utf-8").splitlines():
            if normalized_package_name(line) in skip_packages:
                continue
            keep.append(line)
        filtered_req.write_text("\n".join(keep) + "\n", encoding="utf-8")
        run([sys.executable, "-m", "pip", "install", "-r", filtered_req])

    if protobuf_needs_pin():
        if "google.protobuf" in sys.modules:
            raise RuntimeError(
                "Colab has already imported google.protobuf, but its installed protobuf "
                "version is outside the range needed by this notebook. Restart the runtime "
                "once, then run this setup cell before any other imports."
            )
        run([sys.executable, "-m", "pip", "install", "--upgrade", "protobuf>=5.29.6,<6.30"])

    run([sys.executable, "-m", "pip", "install", "-U", "uv"])

    # Current Colab GPU runtimes ship CUDA 12.8-era PyTorch wheels. Installing
    # torch and vLLM in one transaction keeps their ABI pins aligned and avoids
    # the old CUDA 13 resolver path without requiring a runtime restart.
    vllm_install_base = [
        sys.executable, "-m", "uv", "pip", "install",
        "--system", "--reinstall", "--no-cache", "--torch-backend=cu128",
        "torch", "torchvision", "torchaudio",
    ]
    exact_vllm = run(vllm_install_base + ["vllm==0.19.0"], check=False)
    if exact_vllm.returncode:
        print("vLLM 0.19.0 was not installable for this Colab runtime; falling back to the latest supported 0.18.x wheel.")
        run(vllm_install_base + ["vllm>=0.14,<0.19"])

    # Torchvision imports Pillow during vLLM/Transformers initialization. Keep
    # Pillow pinned so Image.py and the compiled _imaging extension come from
    # the same wheel in Colab's live kernel.
    run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir", "pillow==11.3.0"])
    for module_name in list(sys.modules):
        if module_name == "PIL" or module_name.startswith("PIL."):
            del sys.modules[module_name]
    importlib.invalidate_caches()

    run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", PLUGIN_SRC])

    # Editable installs write .pth metadata that is normally consumed at interpreter
    # startup. Make the source tree importable immediately in this live kernel.
    plugin_path = str(PLUGIN_SRC)
    if plugin_path not in sys.path:
        sys.path.insert(0, plugin_path)
    importlib.invalidate_caches()
    os.chdir(REPO_DIR / "notebooks")

    import torch
    import vllm
    import vllm_hook_plugins

    print(f"torch {torch.__version__} (CUDA runtime: {torch.version.cuda})")
    print(f"vLLM {vllm.__version__}")
    print(f"vllm_hook_plugins loaded from {Path(vllm_hook_plugins.__file__).parent}")
    print("Setup complete. Continue with the next cell; no runtime restart is needed.")
else:
    print("Not running in Colab; install dependencies from the repository README if needed.")
