# 🪝 vLLM-Hook MIA
*A modular plugin library for vLLM.*

📄 [Preprint] [**vLLM Hook** v0: A Plug-in for Programming Model Internals on vLLM](https://arxiv.org/abs/2603.06588v1)

📄 [Preprint] [**MIA**: Runtime-Reconfigurable Access to Model Internals in LLM Inference Engines](./docs/MIA_Arxiv_.pdf)

MIA is a plugin library designed to let developers and researchers **inspect**, **analyze**, and **steer** the internal operations of large language models running under the **vLLM** inference engine.  

This includes dynamic analysis of:  
- attention patterns  
- attention heads  
- activations  
- custom intervention behaviors  

MIA's original capture and steering implementation targets vLLM's V2 PyTorch runner. This
repository is being moved to vLLM-Metal for Apple Silicon. With `VLLM_PLUGINS=metal`, `MiaLLM`
continues to expose normal vLLM generation when constructed without a MIA worker. Capture,
analysis from live captures, steering, and the MIA server plugin are not yet implemented for the
Metal runner; these paths now fail explicitly where the offline API can detect the configuration.

New here? Start with the [Quickstart](#-quickstart).

---

## 📰 News & Events

- **July 24, 2026** — Featured in IBM Think: [*A new way of debugging open-weight models*](https://www.ibm.com/think/news/new-way-debugging-open-weight-models).

- **July 6, 2026** — Presented at ICML 2026: [*vLLM-Hook: Live Programming of Model Internals on vLLM*](https://icml.cc/virtual/2026/75729). (ICML registration and login are required to view the presentation.)

---

## 🚀 Features

- **vLLM generation** through `MiaLLM` on the selected vLLM backend
- Existing analyzers, artifact formats, capture workers, and steering API are retained for the
  original PyTorch runner; Metal instrumentation support is not claimed
- **Example applications**:  
  - Safety guardrails  
  - Reranking  
  - Enhanced instruction following  

---

## 📊 Performance

The following historical CUDA measurements describe the original implementation only; they are
not Metal benchmarks or evidence of Metal instrumentation performance. They were measured on
vLLM 0.29.0 + the V2 runner, server path, K=3 median, against the native vLLM tier of the same
launch (TPOT overhead):

| req/s | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| Hidden states | +4.9 % | +6.1 % | +6.4 % | +5.8 % |
| Q/K | +6.7 % | +8.3 % | +8.4 % | +7.9 % |
| Steering | +3.9 % | +4.1 % | +3.7 % | +5.6 % |

Capture saturates at rate 32; steering holds to 64 (+5.2 %).

---

## 🧩 Supported Configurations

Metal generation runs through vLLM-Metal's MLX-backed model runner. That runner is not
vLLM's V2 GPU/CPU model runner: it does not expose the PyTorch module tree, input-batch hooks,
CUDA graph dispatch, or CUDA custom-op path expected by MIA's current capture workers.
Consequently, MIA capture workers (`capture_hs`, `capture_qk`), activation steering, aperture
delivery, and server-side MIA probes are unsupported on Metal. The established public Python
classes remain available, and the offline generation-only `MiaLLM` path is supported.

---

## 📦 Installation

**Requirements**
- Apple Silicon and a supported vLLM-Metal installation.
- Python version and model support are determined by the installed vLLM-Metal release.
- Unified-memory requirements depend on the model and its context length.
- Gated models require Hugging Face authentication.

### 1. Clone the repository

```bash
git clone https://github.com/IBM/vLLM-Hook.git
cd ./vLLM-Hook
```

### 2. Create an environment and install

Install this repository with the Metal extra in a vLLM-Metal-supported Apple Silicon environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[metal]'
```

The dependency pin follows the supplied vLLM-Metal source version (`0.2.0`); the target package
provides the matching vLLM and MLX runtime dependencies. Select the Metal plugin and the requested
runtime defaults before starting Python:

```bash
export VLLM_PLUGINS=metal
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_METAL_USE_PAGED_ATTENTION=0
export VLLM_METAL_MEMORY_FRACTION=auto
```

---

## ⚡ Quickstart

Generation-only offline use (set `MODEL` to a model identifier or local checkpoint path):

```python
import os
from vllm import SamplingParams
from mia import MiaLLM

llm = MiaLLM(model=os.environ["MODEL"])
outputs = llm.generate(
    "Hello",
    SamplingParams(temperature=0.0, max_tokens=16),
)
print(outputs[0].outputs[0].text)
```

Equivalent standard vLLM use is also supported:

```bash
MODEL=/path/to/model python -c 'from vllm import LLM, SamplingParams; import os; print(LLM(os.environ["MODEL"]).generate("Hello", SamplingParams(max_tokens=16))[0].outputs[0].text)'
```

Do not pass `worker_name="capture_hs"`, `"capture_qk"`, or `"steer"` to `MiaLLM` under
`VLLM_PLUGINS=metal`: these features are unsupported and the offline API raises
`MiaConfigurationError` rather than pretending instrumentation ran. `vllm serve` generation
works with the Metal plugin, but MIA's `MiaClient` probe/capture endpoints are not available.

---

## 👉 Usage

The included capture and steering demos target the original PyTorch runner and are not Metal
compatible. Their commands below are historical usage only; use the generation-only quickstart
above with vLLM-Metal. The use-case documentation likewise does not imply Metal support.

```bash
python examples/demo_hiddenstate.py
```

Each other demo (bar `demo_capture_aperture.py`) also keeps its `vllm serve` version as a
commented block, with the server command to start; `examples/demo_actsteer_serve.py` is the server example
([server mode](examples/README.md#8-server-mode)). [`examples/README.md`](examples/README.md) is
the walkthrough — getting started,
[where captured data lands](examples/README.md#where-the-captured-data-goes), how to confirm a
capture actually happened, and tensor parallelism.
For the full list of use cases see [`docs/use_cases/`](docs/use_cases/README.md).

### Use cases

| Use case | Demo |
|---|---|
| Attention Tracker (in-model safety guardrail) | `python examples/demo_attntracker.py` |
| Core Reranker (in-model relevance ranking) | `python examples/demo_corer.py` |
| Activation Steering (enhanced instruction following) | `python examples/demo_actsteer.py` |
| Hidden-State Probe | `python examples/demo_hiddenstate.py` |
| Science Hallucination Detector | `python examples/demo_scihal.py` |
| H-Node Hallucination Detector | `python examples/demo_halludetect.py` |
| AttnLink-U (schema linking) | `python examples/demo_attnlink.py` |

More demos (language steering, the server-only steering example, the capture-aperture checks,
long decodes): [examples/README.md](examples/README.md#7-running-the-included-demos).

You can customize model configurations in the `model_configs/` folder, e.g.:

```
model_configs/<example_name>/<model_name>.json
```
For example `model_configs/attention_tracker/granite-3.1-8b-instruct.json`.

---

## 🏠 Plugin Architecture

The main package is structured as follows:

```
mia/
├── analyzers/
│   ├── attention_tracker_analyzer.py
│   ├── core_reranker_analyzer.py
├── workers/
│   ├── qk_capture_worker.py
│   ├── steer_worker.py
├── core/
│   ├── runner.py
│   ├── hooks/        (install.py, ops.py)
│   ├── aperture/     (capture_aperture.py, aperture_gather.py)
│   ├── delivery/     (delivery_router.py, artifact_writer.py)
│   ├── runtime/      (tp_shard.py)
├── utils/
│   ├── hnode/
├── artifacts.py
├── client.py
├── llm.py
├── optimizations.py
├── registry.py
```

Each component handles a key stage of the plugin lifecycle:

- **Registry** — manages available hooks and extensions  
- **Workers** — define execution behavior and orchestration  
- **Analyzers** — optionally conduct analysis based on the saved statistics  
- **Core** — the original capture and steering engine, in four subpackages  
  - `hooks/` — installs the PyTorch capture and steering ops  
  - `aperture/` — the GPU capture aperture under CUDA graphs  
  - `delivery/` — routes, writes and delivers captured artifacts  
  - `runtime/` — CPU budget, child-process, device and TP-shard helpers  
- **Utils** — use-case helpers; wraps `hnode/`, the H-Node probe scorer  
- **Runner** — `mia/core/runner.py`, the one place that touches vLLM's V2 model-runner internals  
- **Optimizations** — the public performance levers (`optimizations.py::PUBLIC_LEVERS`)  

Full detail: [`mia/README.md`](mia/README.md).


---

## Repository translation workflow

Translate the repository between declared backends with or without supplied
source and target dependency graphs, save artifacts and open a draft pull request.
Example-specific validation is optional and runs only after generation, outside
the translator's inputs. See the [workflow setup and review instructions](.github/translation/README.md)
for action inputs, local `act` usage, runner dependencies and W&B tracking.

## 🤝 Contributing

We welcome contributions from the community!  

### To contribute:
1. **Fork** this repository  
2. **Create a branch** (`git checkout -b feature/amazing-feature`)  
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)  
4. **Push** to your branch (`git push origin feature/amazing-feature`)  
5. **Open a Pull Request**  

### Guidelines:
- New analyzers and workers are welcome; discuss before modifying `mia/llm.py`, `mia/core/_plugin.py`, `mia/client.py` or `mia/core/`
- Include examples and documentation for new features  
- New use cases must be added to [`docs/use_cases/README.md`](docs/use_cases/README.md) with the contributor's GitHub handle

### Adding a worker or analyzer

- **Analyzer:** a class with `__init__(hook_dir, layer_to_heads)` and
  `analyze(analyzer_spec, run_id=None, probes=None)`; register it with
  `PluginRegistry.register_analyzer("name", Cls)` and pass `analyzer_name="name"`.
  `mia/analyzers/hidden_states_analyzer.py` is the smallest example.
- **Worker:** MIA's three original workers bake their own op into the CUDA graph. A new worker class
  installs PyTorch forward hooks, which only run in eager mode: build its engine with
  `enforce_eager=True` (`--enforce-eager`).

---

## 🌟 Feeling Inspired
```
@article{ko2026vllm,
  title={vLLM Hook v0: A Plug-in for Programming Model Internals on vLLM},
  author={Ko, Ching-Yun and Chen, Pin-Yu},
  journal={arXiv preprint arXiv:2603.06588},
  year={2026}
}
```
---


## IBM ❤️ Open Source AI

vLLM-Hook MIA was started by IBM Research.
- Built for the **vLLM** ecosystem  
- Inspired by community efforts to make LLMs more interpretable and controllable
