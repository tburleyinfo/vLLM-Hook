__all__ = [
    "ProbeHookQKWorkerMetal",
    "ProbeHiddenStatesWorkerMetal",
    "SteerHookActWorkerMetal",
    "SpotlightWorkerMetal",
]


def __getattr__(name):
    if name == "ProbeHookQKWorkerMetal":
        from vllm_hook_plugins.workers.metal.probe_hookqk_worker_metal import (
            ProbeHookQKWorkerMetal,
        )

        return ProbeHookQKWorkerMetal
    if name == "ProbeHiddenStatesWorkerMetal":
        from vllm_hook_plugins.workers.metal.probe_hidden_states_worker_metal import (
            ProbeHiddenStatesWorkerMetal,
        )

        return ProbeHiddenStatesWorkerMetal
    if name == "SteerHookActWorkerMetal":
        from vllm_hook_plugins.workers.metal.steer_activation_worker_metal import (
            SteerHookActWorkerMetal,
        )

        return SteerHookActWorkerMetal
    if name == "SpotlightWorkerMetal":
        from vllm_hook_plugins.workers.metal.spotlight_worker_metal import (
            SpotlightWorkerMetal,
        )

        return SpotlightWorkerMetal
    raise AttributeError(name)
