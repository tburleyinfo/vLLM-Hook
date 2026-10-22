"""Optional native-API validation: source and translated repositories use the same MIA API."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cpu", "target"], required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "blocked", "backend": args.backend, "stage": "imports"}
    engine = None
    try:
        sys.path.insert(0, str(args.candidate.resolve()))
        import torch
        from transformers import AutoTokenizer
        settings = json.loads(args.manifest.read_text())["fixture"]
        report["settings"] = settings
        report["versions"] = {}
        for name in ("vllm", "vllm-metal", "torch", "transformers", "mlx"):
            try:
                report["versions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        mp.set_start_method("spawn", force=True)
        os.environ.setdefault("MIA_WRITER_PROCESS", "0")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        report["stage"] = "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(settings["model"], revision=settings["model_revision"], local_files_only=True)
        token_ids, ranges = [], []
        for prompt in settings["prompts"]:
            encoded = tokenizer(prompt, return_offsets_mapping=True)
            boundary = prompt.index("Data:")
            split = next(i for i, (start, end) in enumerate(encoded["offset_mapping"]) if end > boundary)
            token_ids.append(encoded["input_ids"])
            ranges.append([[0, split], [split, len(encoded["input_ids"])]])
        report.update(token_ids=token_ids, input_ranges=ranges,
                      tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest())
        report["stage"] = "engine"
        from vllm import SamplingParams
        from mia import MiaLLM, register_plugins
        register_plugins()
        config = args.output.parent / "capture_config.json"
        config.write_text(json.dumps({"params": {"important_heads": settings["important_heads"]}, "hookq": {"hookq_mode": settings["hookq_mode"]}}))
        engine = MiaLLM(model=settings["model"], revision=settings["model_revision"],
                        tokenizer_revision=settings["model_revision"], worker_name="capture_qk",
                        analyzer_name="attn_tracker", config_file=str(config),
                        hook_dir=str(args.output.parent / "hooks"), dtype=settings["dtype"],
                        max_model_len=settings["max_model_len"], seed=settings["seed"],
                        max_num_seqs=settings["max_num_seqs"],
                        max_num_batched_tokens=settings["max_num_batched_tokens"],
                        kv_cache_memory_bytes=settings["kv_cache_memory_bytes"],
                        enable_prefix_caching=False, enforce_eager=True, enable_hook=True,
                        gpu_memory_utilization=0.25)
        report["stage"] = "generate"
        outputs = engine.generate(settings["prompts"], SamplingParams(temperature=settings["temperature"], max_tokens=settings["max_tokens"]), use_hook=True)
        probes = outputs[0].probes
        report["generate_executed"] = True
        report["stage"] = "analyze"
        stats = engine.analyze(analyzer_spec={"input_range": ranges, "attn_func": settings["attn_func"]}, probes=probes)
        report["analyze_executed"] = True
        from mia.artifacts import unpack_qk
        captures = []
        config = probes["config"]
        assert config["num_attention_heads"] > 0 and config["num_key_value_heads"] > 0
        assert config["head_dim"] > 0
        for name, entry in probes["qk_cache"].items():
            q_list, k_list = unpack_qk(entry)
            assert len(q_list) == len(k_list) == len(settings["prompts"])
            for request, (q, k) in enumerate(zip(q_list, k_list)):
                q, k = torch.as_tensor(q), torch.as_tensor(k)
                assert q.numel() == config["num_attention_heads"] * config["head_dim"]
                assert list(k.shape) == [len(token_ids[request]), config["num_key_value_heads"] * config["head_dim"]]
                assert torch.isfinite(q).all() and torch.isfinite(k).all()
                assert entry["hookq_mode"] == settings["hookq_mode"]
                captures.append({"request": request, "layer": int(entry["layer_num"]),
                                 "q_shape": list(q.shape), "k_shape": list(k.shape),
                                 "q_dtype": str(q.dtype), "k_dtype": str(k.dtype), "finite": True})
        assert captures and {c["layer"] for c in captures} == {h[0] for h in settings["important_heads"]}
        scores = [float(v) for v in stats["score"]]
        assert len(scores) == len(settings["prompts"])
        assert all(torch.isfinite(torch.tensor(scores)))
        report.update(status="pass", stage="complete", scores=scores, captures=captures,
                      capture_config=config, output_count=len(outputs))
        assert len(outputs) == len(settings["prompts"])
    except Exception as exc:
        import traceback
        report.update(status="blocked" if report["stage"] in ("imports", "tokenizer", "engine") else "fail",
                      error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
        if engine is not None:
            if hasattr(engine, "close"):
                engine.close()
            elif hasattr(engine, "llm_engine"):
                engine.llm_engine.engine_core.shutdown()
    print(json.dumps({"backend": args.backend, "status": report["status"], "stage": report["stage"]}))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
