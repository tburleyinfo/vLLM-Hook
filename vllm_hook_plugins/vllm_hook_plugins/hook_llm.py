import os
import json
import glob
import uuid
import platform
import importlib.util
from typing import Optional, Dict, List
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from vllm import LLM, SamplingParams


def _use_metal_workers() -> bool:
    if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
        return False
    return importlib.util.find_spec("vllm_metal") is not None


class HookLLM:
    def __init__(
        self,
        model: str,
        worker_name: str = None,
        analyzer_name: str = None,
        config_file: str = None,
        # Original: download_dir: str = '~/.cache',
        download_dir: str = "~/.cache/vllm_hook",
        enable_hook: bool = True,
        hook_dir: str = None,
        enforce_eager: bool = True,
        **vllm_kwargs
    ):

        self.model_name = model
        self.worker_name = worker_name
        self.analyzer_name = analyzer_name
        self.enable_hook = enable_hook
        self.enforce_eager = enforce_eager

        # Original:
        # if hook_dir is not None:
        #     HOOK_DIR = hook_dir
        # else:
        #     HOOK_DIR = os.path.join(download_dir,'_v1_qk_peeks')
        # os.makedirs(HOOK_DIR, exist_ok=True)
        # self._hook_dir = HOOK_DIR
        resolved_download_dir = os.path.abspath(os.path.expanduser(download_dir))
        if hook_dir is not None:
            resolved_hook_dir = os.path.abspath(os.path.expanduser(hook_dir))
        else:
            resolved_hook_dir = os.path.join(resolved_download_dir, "_v1_qk_peeks")

        os.makedirs(resolved_hook_dir, exist_ok=True)
        self._hook_dir = resolved_hook_dir
        self._hook_flag = os.path.join(self._hook_dir, "EXTRACT.flag")
        self._run_id_file = os.path.join(self._hook_dir, "RUN_ID.txt")

        os.environ["VLLM_HOOK_DIR"] = os.path.abspath(self._hook_dir)
        os.environ["VLLM_HOOK_FLAG"] = os.path.abspath(self._hook_flag)
        os.environ["VLLM_RUN_ID"] = os.path.abspath(self._run_id_file)

        self.layer_to_heads = {}
        if config_file:
            self.load_config(config_file)

        if _use_metal_workers():
            # Avoid c10d TCPStore/bootstrap hangs on macOS by pinning loopback
            # and using single-process engine startup by default.
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
            os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")

        #TODO: Backend Negotiation needs to happen here.
        worker = None
        if worker_name:
            import vllm.plugins
            from vllm_hook_plugins import PluginRegistry
            vllm.plugins.load_general_plugins()

            # Original: worker = PluginRegistry.get_worker(worker_name).path
            worker_obj = PluginRegistry.get_worker(worker_name)
            if worker_obj is None:
                raise ValueError(f"Unknown worker_name: {worker_name}")

            worker = worker_obj.path
            if _use_metal_workers():
                if worker_name == "probe_hook_qk":
                    from vllm_hook_plugins.workers.metal import ProbeHookQKWorkerMetal
                    worker = f"{ProbeHookQKWorkerMetal.__module__}.{ProbeHookQKWorkerMetal.__name__}"
                elif worker_name == "steer_hook_act":
                    from vllm_hook_plugins.workers.metal import SteerHookActWorkerMetal
                    worker = f"{SteerHookActWorkerMetal.__module__}.{SteerHookActWorkerMetal.__name__}"


        self.llm = LLM(
            model=model,
            # Original: download_dir=download_dir,
            download_dir=resolved_download_dir,
            worker_cls=worker,
            enforce_eager = enforce_eager,
            **vllm_kwargs
        )

        self.tokenizer = self.llm.get_tokenizer()
        self.llm_engine = self.llm.llm_engine

        self.analyzer = None
        if analyzer_name:
            self.analyzer = PluginRegistry.get_analyzer(analyzer_name).analyzer
            self.analyzer = self.analyzer(self._hook_dir, self.layer_to_heads)


    def load_config(self, config_file: str):
        with open(config_file, 'r') as f:
            config_data = json.load(f)

        if "params" in config_data and "important_heads" in config_data["params"]:
            self.important_heads = config_data["params"]["important_heads"]
            # self.important_heads = [[i, j] for i in range(32) for j in range(32)]
            self.layer_to_heads = {}
            for layer_idx, head_idx in self.important_heads:
                if layer_idx not in self.layer_to_heads:
                    self.layer_to_heads[layer_idx] = []
                self.layer_to_heads[layer_idx].append(head_idx)

            layer_to_heads_string = ";".join([
                f"{layer}:{','.join(map(str, heads))}"
                for layer, heads in sorted(self.layer_to_heads.items())
            ])
            os.environ["VLLM_HOOK_LAYER_HEADS"] = layer_to_heads_string

        if "hookq" in config_data:
            hookq_mode = config_data["hookq"]["hookq_mode"]
            os.environ["VLLM_HOOKQ_MODE"] = hookq_mode

        if "steering" in config_data:
            os.environ["VLLM_ACTSTEER_CONFIG"] = os.path.abspath(config_file)

    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        use_hook: Optional[bool] = None,
        cleanup: Optional[bool] = True,
        **kwargs
    ):
        hook = use_hook if use_hook is not None else self.enable_hook

        if not isinstance(prompts, list):
            prompts = [prompts]

        if hook:
            if "probe" in self.worker_name:
                return self.generate_with_encode_hook(prompts, sampling_params, cleanup, **kwargs)
            elif "steer" in self.worker_name:
                return self.generate_with_decode_hook(prompts, sampling_params, cleanup, **kwargs)

        else:
            if sampling_params is None:
                sampling_params = SamplingParams(**kwargs)
            return self.llm.generate(prompts, sampling_params)

    def generate_with_encode_hook(self, prompts, sampling_params, cleanup, **kwargs):

        self._setup_hooks(cleanup)

        # prefill with hooks
        prefill_params = SamplingParams(temperature=0.1, max_tokens=1)
        self.llm.generate(prompts, prefill_params)

        self._cleanup_hooks()
        output = None
        # generation without hooks
        if sampling_params is None:
            sampling_params = SamplingParams(**kwargs)
        output = self.llm.generate(prompts, sampling_params)

        return output

    def generate_with_decode_hook(self, prompts, sampling_params, cleanup, **kwargs):

        # prefill without hooks
        prefill_params = SamplingParams(temperature=0.1, max_tokens=1)
        self.llm.generate(prompts, prefill_params)

        self._setup_hooks(cleanup)

        # generation with hooks
        if sampling_params is None:
            sampling_params = SamplingParams(**kwargs)
        output = self.llm.generate(prompts, sampling_params)

        self._cleanup_hooks()

        return output

    def analyze(
        self,
        analyzer_spec: Optional[Dict] = None
    ) -> Optional[Dict]:

        if self.analyzer is None:
            print("No analyzer configured")
            return None

        return self.analyzer.analyze(analyzer_spec)


    def _setup_hooks(self, cleanup):
        if cleanup:
            for p in glob.glob(os.path.join(self._hook_dir, "**", "qk.pt"), recursive=True):
                os.remove(p)
                print("Cleaned up previous qk cache.")
            if os.path.exists(self._run_id_file):
                os.remove(self._run_id_file)

        run_id = str(uuid.uuid4())
        with open(self._run_id_file, "a") as f:
            f.write(run_id+ "\n")
            print("Logged run ID.")

        open(self._hook_flag, "a").close()
        print("Created hook flag.")


    def _cleanup_hooks(self):
        if os.path.exists(self._hook_flag):
            os.remove(self._hook_flag)
            print("Hooks deactivated.")
        else:
            print("No hooks to be deactivated.")
