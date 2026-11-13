import sys
import types

import pytest

from mia.core.backends.vllm_metal import MetalQKTap


class _Attention:
    pass


class _Layer:
    def __init__(self):
        self.self_attn = _Attention()


class _Model:
    def __init__(self):
        self.layers = [_Layer(), _Layer()]


def _install_fake_metal(monkeypatch):
    sdpa = types.ModuleType("vllm_metal.attention.impls.sdpa")
    result = ("q", "k", "v", None, ("k", "v"))

    def prepare(inner, x, ctx, n_heads, n_kv_heads, *args, **kwargs):
        return result

    sdpa.prepare_sdpa_qkv = prepare
    patching = types.ModuleType("vllm_metal.attention.patching")
    patching.find_layers = lambda model: model.layers
    for name in (
        "vllm_metal",
        "vllm_metal.attention",
        "vllm_metal.attention.impls",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules, "vllm_metal.attention.impls.sdpa", sdpa
    )
    monkeypatch.setitem(
        sys.modules, "vllm_metal.attention.patching", patching
    )
    return sdpa, result


def test_tap_observes_qk_only_in_capture_scope_and_preserves_return(monkeypatch):
    sdpa, original_result = _install_fake_metal(monkeypatch)
    model = _Model()
    seen = []
    tap = MetalQKTap(model, lambda *args: seen.append(args))
    result = sdpa.prepare_sdpa_qkv(
        model.layers[1].self_attn, None, "context", 1, 1
    )
    assert result is original_result
    assert seen == []

    with tap.capture():
        result = sdpa.prepare_sdpa_qkv(
            model.layers[1].self_attn, None, "context", 1, 1
        )

    assert result is original_result
    assert seen == [(1, "context", "q", "k")]
    tap.close()
    assert sdpa.prepare_sdpa_qkv.__name__ == "prepare"


def test_tap_requires_attention_layers_and_callable_observer(monkeypatch):
    _install_fake_metal(monkeypatch)
    with pytest.raises(TypeError, match="observer"):
        MetalQKTap(_Model(), None)
    with pytest.raises(ValueError, match="No SDPA"):
        MetalQKTap(types.SimpleNamespace(layers=[]), lambda *args: None)


def test_tap_restores_original_function_when_scope_raises(monkeypatch):
    sdpa, _ = _install_fake_metal(monkeypatch)
    original = sdpa.prepare_sdpa_qkv
    tap = MetalQKTap(_Model(), lambda *args: None)
    with pytest.raises(ZeroDivisionError):
        with tap:
            1 / 0
    assert sdpa.prepare_sdpa_qkv is original
