"""Generic live attention observation. No Q/K reconstruction or tracker logic."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True

from live_precision import align_live_precision
from request_identity import capture_request_mapping, internal_id


def describe(value):
    cls = type(value)
    path = inspect.getsourcefile(cls)
    result = {"class": cls.__module__ + "." + cls.__qualname__}
    if path:
        file = Path(path).resolve()
        result.update(source_file=str(file), source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                      source_line=inspect.getsourcelines(cls)[1])
    return result


class ObservationExtension:
    def observation_install(self, layer_index):
        import mlx.nn as nn
        from vllm_metal.paged_attention_common import find_layers, find_attn_attr
        precision = align_live_precision(self)
        model = self.model_runner.model
        layer = find_layers(model)[layer_index]
        attribute = find_attn_attr(layer)
        if attribute is None:
            raise RuntimeError("Live layer has no recognized attention callable")
        original = getattr(layer, attribute)
        observations = []
        class Wrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def __call__(self, *args, **kwargs):
                observations.append({"input_shapes": [list(x.shape) for x in args if hasattr(x, "shape")],
                                     "keyword_names": sorted(kwargs)})
                return self.module(*args, **kwargs)
        self._observation_state = (layer, attribute, original, observations)
        setattr(layer, attribute, Wrapper(original))
        chain = []
        component = original
        for _ in range(4):
            chain.append(describe(component))
            inner = getattr(component, "_inner", None)
            if inner is None or inner is component:
                break
            component = inner
        return {"precision": precision, "runner": describe(self.model_runner), "model": describe(model),
                "layer": describe(layer), "attention_chain": chain, "attention_attribute": attribute,
                "projection_attributes": {name: hasattr(component, name) for name in ("q_proj", "k_proj", "v_proj", "rope")}}

    def observation_collect(self):
        return self._observation_state[3]

    def observation_restore(self):
        layer, attribute, original, observations = self._observation_state
        setattr(layer, attribute, original)
        restored = getattr(layer, attribute) is original
        self._observation_state = None
        return restored


def main():
    manifest, output = Path(sys.argv[1]), Path(sys.argv[2])
    settings = json.loads(manifest.read_text())["fixture"]
    engine = None
    report = {"status": "fail", "limits": "Generic live-call observation only; Q/K reconstruction and parity remain unverified."}
    try:
        from vllm import LLM, SamplingParams
        engine = LLM(model=settings["model"], revision=settings["model_revision"],
                     tokenizer_revision=settings["model_revision"], dtype=settings["dtype"],
                     max_model_len=settings["max_model_len"], max_num_seqs=settings["max_num_seqs"],
                     max_num_batched_tokens=settings["max_num_batched_tokens"],
                     kv_cache_memory_bytes=settings["kv_cache_memory_bytes"],
                     enable_prefix_caching=False, seed=settings["seed"], enforce_eager=True,
                     worker_extension_cls="live_worker_probe.ObservationExtension")
        rpc = engine.llm_engine.collective_rpc
        report["workers"] = rpc("observation_install", args=(settings["important_heads"][0][0],))
        params = SamplingParams(temperature=settings["temperature"], max_tokens=settings["max_tokens"])
        with capture_request_mapping(engine.llm_engine, "wrapped-request-id-map.json") as mapping:
            wrapped = engine.generate(settings["prompts"], params)
        report["request_identity"] = {str(o.request_id): internal_id(mapping, o.request_id) for o in wrapped}
        report["observations"] = rpc("observation_collect")
        report["restored"] = rpc("observation_restore")
        with capture_request_mapping(engine.llm_engine, "plain-request-id-map.json") as plain_mapping:
            plain = engine.generate(settings["prompts"], params)
        report["plain_request_identity"] = {str(o.request_id): internal_id(plain_mapping, o.request_id) for o in plain}
        tokens = lambda outputs: [list(o.outputs[0].token_ids) for o in outputs]
        report["wrapped_output_token_ids"] = tokens(wrapped)
        report["plain_output_token_ids"] = tokens(plain)
        assert all(report["restored"]), "Wrapper restoration failed"
        assert all(report["observations"]), "No live attention calls observed"
        assert tokens(wrapped) == tokens(plain), "Wrapper changed deterministic output tokens"
        if any(w["precision"]["after"]["parameter_dtype_counts"] != {"mlx.core.float32": w["precision"]["after"]["parameter_count"]} for w in report["workers"]):
            raise RuntimeError("Live model contains parameters outside float32")
        report.update(precision_verified=True, request_identity_verified=True, status="pass", generate_executed=True, output_tokens_unchanged=True, settings=settings)
    except Exception:
        report.update(error=traceback.format_exc())
        traceback.print_exc()
    finally:
        if engine is not None:
            try:
                engine.llm_engine.engine_core.shutdown()
            except RuntimeError as error:
                # The installed vLLM calls Torch accelerator cleanup after the
                # Metal worker shuts down. Torch 2.11 rejects its MPS allocator.
                if "Allocator for mps is not a DeviceAllocator" not in str(error):
                    report.update(status="fail", shutdown_error=traceback.format_exc())
                else:
                    try:
                        import gc
                        import torch
                        import mlx.core as mx
                        gc.collect()
                        torch.mps.empty_cache()
                        mx.clear_cache()
                        report["cleanup_warning"] = str(error)
                        report["cleanup_fallback"] = "MPS-specific empty_cache and MLX clear_cache after worker shutdown"
                    except Exception:
                        report.update(status="fail", shutdown_error=traceback.format_exc())
            except Exception:
                report.update(status="fail", shutdown_error=traceback.format_exc())
        output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
