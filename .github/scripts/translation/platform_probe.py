"""Small platform witnesses, not an Attention Tracker implementation."""
import json
import sys
import traceback


def main():
    results = {}
    def check(name, function):
        try:
            results[name] = {"status": "pass", "observation": function()}
        except Exception:
            results[name] = {"status": "fail", "error": traceback.format_exc()}

    def wrapping():
        import mlx.core as mx
        import mlx.nn as nn
        events = []
        class Wrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def __call__(self, x):
                events.append("before")
                y = self.inner(x)
                events.append("after")
                return y
        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(2, 2)
            def __call__(self, x):
                return self.layer(x)
        parent = Parent()
        original = parent.layer
        x = mx.array([[1., 2.]])
        expected = parent(x)
        names = [name for name, _ in parent.named_modules()]
        parent.layer = Wrapper(original)
        actual = parent(x)
        mx.eval(expected, actual)
        assert bool(mx.allclose(expected, actual).item())
        assert events == ["before", "after"]
        parent.layer = original
        assert parent.layer is original
        return {"module_names": names, "callbacks": events, "equal_output": True, "restored": True}

    def bridge():
        import mlx.core as mx
        from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
        x = mx.array([[1., 2.]], dtype=mx.float32)
        mx.eval(x)
        y = mlx_to_torch(x).cpu()
        assert y.tolist() == [[1., 2.]]
        return {"shape": list(y.shape), "dtype": str(y.dtype), "values_match": True}

    check("mlx_module_wrapping", wrapping)
    check("mlx_to_torch", bridge)
    with open(sys.argv[1], "w") as output:
        json.dump({"status": "pass" if all(x["status"] == "pass" for x in results.values()) else "fail",
                   "probes": results,
                   "limits": "Toy modules only. Worker access, request identity, Q/K timing and real generation remain unverified."}, output, indent=2)
    return 0 if all(x["status"] == "pass" for x in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
