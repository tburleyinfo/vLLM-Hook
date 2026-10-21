"""Bounded discovery and immutable shared evidence for MLR-15/40."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import repository_graph

FILES = (
    "vllm_metal/v1/worker.py", "vllm_metal/v1/model_runner.py",
    "vllm_metal/v1/model_lifecycle.py", "vllm_metal/v1/model_adapter.py",
    "vllm_metal/paged_attention_common.py", "vllm_metal/paged_attention_backend/mha.py",
    "vllm_metal/pytorch_backend/tensor_bridge.py", "mlx_lm/models/qwen2.py",
    "vllm_metal/metal_kernel_backend/paged_attention.py",
    "vllm_metal/paged_attention_backend/hybrid.py", "vllm_metal/paged_attention_backend/mla.py",
    "mlx_lm/models/base.py", "mlx_lm/models/cache.py", "mlx/nn/layers/base.py",
)
REQUIREMENTS = (
    ("worker_model", "V2 runner is required", "Access the live target model in its worker", ("model_runner", "self.model")),
    ("module_observation", "PyTorch forward hooks", "Observe and restore a target module call", ("named_modules", "__call__", "update_modules")),
    ("post_rope_qk", "Source Q/K capture hook points", "Capture Q/K after positional transforms without changing attention", ("q_proj", "k_proj", "rope", "scaled_dot_product_attention")),
    ("request_delivery", "Source probe delivery", "Associate prompt captures with request IDs", ("request_id", "req_id", "execute_model")),
    ("tensor_bridge", "Torch analyzer inputs", "Convert captured target arrays to CPU Torch tensors", ("mlx_to_torch",)),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def discover(run, site):
    destination = run / "platform-discovery"
    if destination.exists():
        raise ValueError("Discovery already exists; use a new run to change target evidence")
    site = Path(site).resolve()
    if not site.is_dir():
        raise ValueError("Provide an installed target site directory")
    index, missing = {}, []
    for name in FILES:
        source = (site / name).resolve()
        if not source.is_relative_to(site):
            raise ValueError("Target source escapes the selected installation")
        if not source.is_file():
            missing.append(name)
            continue
        if source.stat().st_size > 1_000_000:
            raise ValueError("Target file exceeds bounded discovery size")
        target = destination / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        index[name] = {"sha256": sha(target), "lines": len(target.read_text().splitlines())}
    if not index:
        raise ValueError("No selected platform files found")
    versions = {}
    for metadata in site.glob("*.dist-info/METADATA"):
        if metadata.parent.name.startswith(("vllm", "mlx", "torch", "transformers")):
            versions[metadata.parent.name] = {"sha256": sha(metadata), "headers": metadata.read_text().split("\n\n", 1)[0]}
    rows = []
    for key, assumption, behavior, terms in REQUIREMENTS:
        hits = []
        for name in index:
            lines = (destination / "source" / name).read_text().splitlines()
            for number, line in enumerate(lines, 1):
                if any(term in line for term in terms):
                    hits.append({"file": name, "line": number, "text": line.strip()})
                    if sum(x["file"] == name for x in hits) >= 12:
                        break
        rows.append({"id": key, "source_assumption": assumption, "required_behavior": behavior,
                     "status": "source_leads_only" if hits else "unknown", "evidence": hits,
                     "next_step": "Inspect surrounding source and validate on real worker; textual matches do not prove capability"})
    save(destination / "index.json", {"files": index, "versions": versions, "missing": missing,
         "target_site": str(site), "holdout": "Completed apple-silicon vLLM-Hook port excluded"})
    save(destination / "capabilities.json", rows)
    shutil.copyfile(Path(__file__).with_name("platform_probe.py"), destination / "probe.py")
    shutil.copyfile(Path(__file__).with_name("live_worker_probe.py"), destination / "live_worker_probe.py")
    for helper in ("live_precision.py", "request_identity.py", "capture_metadata.py"):
        shutil.copyfile(Path(__file__).with_name(helper), destination / helper)
    save(run / "discovery-result.json", {"status": "pass", "method": "bounded deterministic source search",
         "files": len(index), "missing": missing, "discovery_sha256": inventory(destination)})


def verify_discovery(run):
    expected = json.loads((run / "discovery-result.json").read_text())["discovery_sha256"]
    index = json.loads((run / "platform-discovery/index.json").read_text())
    site = Path(index["target_site"])
    for name, record in index["files"].items():
        source = (site / name).resolve()
        if not source.is_relative_to(site) or not source.is_file() or sha(source) != record["sha256"]:
            raise ValueError("Installed target source changed after discovery")
    actual = inventory(run / "platform-discovery")
    if actual != expected:
        raise ValueError("Discovery evidence changed after collection")


def freeze(run):
    verify_discovery(run)
    probes = json.loads((run / "platform-probe-result.json").read_text())
    if probes.get("status") != "pass":
        raise ValueError("Platform witnesses must pass before freezing translation context")
    cpu = json.loads((run / "cpu-result.json").read_text())
    if cpu.get("status") != "pass":
        raise ValueError("Shared CPU baseline must pass before freezing context")
    manifest = json.loads((run / "manifest.json").read_text())
    live = manifest.get("protocol_version", 1) >= 3
    if live:
        verify_live(run)
    if manifest.get("protocol_version", 1) >= 4:
        report = json.loads((run / "live-worker-result.json").read_text())
        if not report.get("precision_verified") or not report.get("request_identity_verified"):
            raise ValueError("Live precision and request identity must be verified before freezing")
    retrieval = manifest.get("protocol_version", 1) >= 5
    if retrieval:
        repository_graph.verify(run)
    context = run / "shared-context"
    if context.exists():
        raise ValueError("Shared context already frozen; use a new run for changes")
    for condition in ("source_only", "source_validated_dag"):
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=run / condition / "candidate").strip():
            raise ValueError("Cannot freeze evidence into an already translated candidate")
    shutil.copytree(run / "platform-discovery", context)
    shutil.copyfile(run / "platform-probe-result.json", context / "probe-result.json")
    (context / "README.md").write_text("Selected target platform source, source search leads, and executable toy witnesses.\n"
        "The completed Apple Silicon Attention Tracker port is held out.\n"
        "Wrapping and conversion witnesses do not establish real worker Q/K capture or parity.\n"
        "Find target primitives that satisfy the behavior; do not require the source runner class or PyTorch hooks.\n")
    if live:
        shutil.copytree(run / "live-platform", context / "live-platform")
        shutil.copyfile(run / "live-worker-result.json", context / "live-worker-result.json")
        (context / "README.md").write_text((context / "README.md").read_text() +
            "Read live-worker-result.json and live-platform/source for the actual loaded attention path.\n"
            "live_worker_probe.py supplies generic worker access and wrapping only. Use its mechanism; adapt delivery for tracker captures.\n"
            "named_modules locates modules, not internal Q/K values. Reconstruct required Q/K from the live attention inputs, projections, positional transforms and cache semantics without mutating the live cache.\n"
            "Use live_precision.py to align actual loaded MLX parameters before generation; requested dtype alone is insufficient.\n"
            "Use request_identity.py to snapshot explicit external/internal IDs around generation; do not strip suffixes. Copy supplied helpers to a runnable location rather than editing platform_reference.\n"
            "capture_metadata.py defines the capture config schema; important_heads comes from tracker settings and hookq_mode belongs to layer entries.\n"
            "The completed upstream reconstruction is held out. A successful observation is not Q/K correctness.\n")
    hashes = inventory(context)
    protocol = json.loads((run / "protocol.json").read_text())
    protocol["retrieval_files"] = inventory(run / "target-retrieval") if retrieval else {}
    protocol["condition_context_files"] = {}
    for condition in ("source_only", "source_validated_dag"):
        directory = run / condition
        candidate = directory / "candidate"
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=candidate).strip():
            raise ValueError("Cannot freeze evidence into an already translated candidate")
        shutil.copytree(context, candidate / "platform_reference")
        if retrieval and condition == "source_validated_dag":
            (candidate / "target_relationships").mkdir()
            for name in ("retrieval.json", "relationships.md"):
                shutil.copyfile(run / "target-retrieval" / name, candidate / "target_relationships" / name)
        protocol["condition_context_files"][condition] = inventory(candidate / "target_relationships")
        subprocess.run(["git", "add", "platform_reference", *(["target_relationships"] if retrieval and condition == "source_validated_dag" else [])], cwd=candidate, check=True)
        subprocess.run(["git", "-c", "user.name=Experiment", "-c", "user.email=experiment@localhost",
                        "-c", "commit.gpgsign=false", "commit", "-qm", "Frozen shared platform evidence"], cwd=candidate, check=True)
        prompt = (directory / "prompt.md").read_text()
        prompt += "\nShared target evidence: read platform_reference/README.md, capabilities.json, probe-result.json and selected source files.\nTreat platform_reference as immutable reference data. Source runner and hook APIs are assumptions to adapt, not target requirements.\n"
        if retrieval and condition == "source_validated_dag":
            prompt += "\nAdditional condition: inspect target_relationships/relationships.md and retrieval.json for static target-library call/definition relationships. These are retrieval leads, not proof of runtime behavior. All underlying source files are identical to the no-graph condition.\n"
        (directory / "prompt.md").write_text(prompt)
        protocol["prompt_sha256"][condition] = hashlib.sha256(prompt.encode()).hexdigest()
    protocol["shared_context_files"] = hashes
    protocol["condition_labels"] = {"source_only": "source + shared platform evidence", "source_validated_dag": "source + shared platform evidence + source DAG"}
    if retrieval:
        protocol["condition_labels"] = {"source_only": "ordinary repository search + identical shared platform evidence",
                                        "source_validated_dag": "same evidence/search + deterministic target-library AST relationships"}
        protocol["retrieval"] = manifest["retrieval"]
        protocol["limits"] = "Graph relationships are the context intervention; source, tools, model settings and timeout match. Token/tool budgets are not enforced, so this is not a token-matched efficiency study."
    save(run / "protocol.json", protocol)
    save(run / "freeze-result.json", {"status": "pass", "files": hashes,
         "live_observation_required": live,
         "limits": "Live module observation passed; Q/K reconstruction and parity remain unverified" if live else probes["limits"]})


def verify_context(run, condition):
    protocol = json.loads((run / "protocol.json").read_text())
    expected = protocol.get("shared_context_files")
    if not expected or inventory(run / "shared-context") != expected:
        raise ValueError("Shared evidence is missing or changed; run discover, probe, freeze first")
    if protocol.get("protocol_version", 1) >= 5:
        repository_graph.verify(run)
        if inventory(run / "target-retrieval") != protocol["retrieval_files"]:
            raise ValueError("Frozen retrieval evidence changed")
        actual = inventory(run / condition / "candidate/target_relationships")
        if actual != protocol["condition_context_files"][condition]:
            raise ValueError("Candidate graph context changed")
    if inventory(run / condition / "candidate/platform_reference") != expected:
        raise ValueError("Candidate shared evidence changed")


def collect_live(run):
    report = json.loads((run / "live-worker-result.json").read_text())
    if report.get("status") != "pass":
        raise ValueError("Live worker observation must pass")
    site = Path(json.loads((run / "platform-discovery/index.json").read_text())["target_site"]).resolve()
    destination = run / "live-platform"
    if destination.exists():
        raise ValueError("Live evidence already exists; use a new run")
    records = []
    for worker in report["workers"]:
        records.extend([worker["runner"], worker["model"], worker["layer"], *worker["attention_chain"]])
    for record in records:
        path = Path(record["source_file"]).resolve()
        if not path.is_relative_to(site) or path.relative_to(site).parts[0] not in ("mlx", "mlx_lm", "vllm_metal"):
            raise ValueError("Live source is outside permitted platform packages")
        if path.suffix != ".py" or path.stat().st_size > 1_000_000 or sha(path) != record["source_sha256"]:
            raise ValueError("Live source changed or exceeds discovery bounds")
    for record in records:
        path = Path(record["source_file"]).resolve()
        target = destination / "source" / path.relative_to(site)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    save(run / "live-evidence-index.json", {"files": inventory(destination), "result_sha256": sha(run / "live-worker-result.json")})


def verify_live(run):
    index = json.loads((run / "live-evidence-index.json").read_text())
    if inventory(run / "live-platform") != index["files"] or sha(run / "live-worker-result.json") != index["result_sha256"]:
        raise ValueError("Live worker evidence changed")
    report = json.loads((run / "live-worker-result.json").read_text())
    if report.get("status") != "pass":
        raise ValueError("Live worker observation must pass")
    for worker in report["workers"]:
        for record in [worker["runner"], worker["model"], worker["layer"], *worker["attention_chain"]]:
            path = Path(record["source_file"])
            if not path.is_file() or sha(path) != record["source_sha256"]:
                raise ValueError("Installed live attention source changed")
