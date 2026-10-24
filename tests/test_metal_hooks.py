from mia.core.hooks.metal import (
    MetalHookRuntime,
    MetalRequest,
    _layer_number,
    _selected_layers,
)


class _Layer:
    def __call__(self, value):
        return value + 1


class _Model:
    def __init__(self):
        self.layer = _Layer()

    def named_modules(self):
        return [("model.layers.0", self.layer)]


def test_metal_layer_lookup_and_selection():
    assert _layer_number("model.layers.2") == 2
    assert _layer_number("model.layers.2.self_attn") is None
    assert _selected_layers([1, "3"], 0)
    assert _selected_layers({2: []}, 1)
    assert not _selected_layers([2], 0)


def test_metal_hook_wrapper_preserves_unarmed_forward_and_closes():
    model = _Model()
    runtime = MetalHookRuntime(model)
    assert model.layer(4) == 5

    request = MetalRequest(
        request_id="r1",
        extra_args={"output_hidden_states": [1]},
        start=0,
        end=2,
        is_prefill=True,
    )
    with runtime.request_scope([request]):
        assert runtime.probes == {}
    runtime.close()
    assert model.layer(4) == 5
    assert not hasattr(model.layer, "_mia_metal_runtime")
