"""Inspect and align actual live MLX parameters, not just capture output casts."""
from pathlib import Path
import json

def align_live_precision(worker, artifact_path="live-precision.json"):
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map
    model = worker.model_runner.model
    def summary():
        values = tree_flatten(model.parameters())
        counts = {}
        for _, value in values:
            counts[str(value.dtype)] = counts.get(str(value.dtype), 0) + 1
        return {"parameter_dtype_counts": counts, "parameter_count": len(values)}
    before = summary()
    model.update(tree_map(lambda x: x.astype(mx.float32) if x.dtype in (mx.float16, mx.bfloat16, mx.float32) else x, model.parameters()))
    mx.eval(model.parameters())
    after = summary()
    if any(dtype in after["parameter_dtype_counts"] for dtype in (str(mx.float16), str(mx.bfloat16))):
        raise RuntimeError("Model precision alignment left low-precision parameters")
    report = {"before": before, "after": after, "scope": "Entire loaded model parameters, before request generation; checkpoint values remain the same"}
    Path(artifact_path).write_text(json.dumps(report, indent=2) + "\n")
    print("[precision] " + json.dumps(report), flush=True)
    return report
