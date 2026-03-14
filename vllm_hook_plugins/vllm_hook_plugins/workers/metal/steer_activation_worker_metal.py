import json
import os
from typing import Dict

import mlx.core as mx
import torch
from vllm_metal.v1.worker import MetalWorker


class SteerHookActWorkerMetal(MetalWorker):
    def load_model(self, *args, **kwargs):
        r = super().load_model(*args, **kwargs)

        try:
            self._install_hooks()
            print("Hooks installed successfully")
        except Exception as e:
            print(f"Hook installation failed: {e}")
            raise

        return r

    def _install_hooks(self):
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        self.hook_flag = os.environ.get("VLLM_HOOK_FLAG")
        steering_config = self._parse_steering_config()
        self.steering_method = steering_config["method"]
        self.optimal_layer = steering_config["optimal_layer"]
        self.coefficient = steering_config["coefficient"]
        self.apply_at_all_positions = steering_config["apply_at_all_positions"]

        vector_path = steering_config["vector_path"]
        if not os.path.exists(vector_path):
            raise FileNotFoundError(f"Steering vector not found at: {vector_path}")
        steering_data = torch.load(vector_path)
        self.dir = mx.array(steering_data["dir"])
        if self.steering_method == "adjust_rs":
            self.avg_proj = mx.array(steering_data["avg_proj"])
            self.unit_vector = self.dir

        def steering_hook(input, output):
            if not os.path.exists(self.hook_flag):
                return output

            residuals = output
            steering_vec = self.dir.astype(residuals.dtype)

            if self.steering_method == "add_vector":
                if not self.apply_at_all_positions:
                    raise NotImplementedError("Only supports apply_at_all_positions=True for now.")
                residuals = residuals + self.coefficient * steering_vec.reshape(1, 1, -1)
            elif self.steering_method == "adjust_rs":
                unit_vec = self.unit_vector.astype(residuals.dtype)
                avg_proj = self.avg_proj.astype(residuals.dtype)
                current_projections = mx.sum(residuals * unit_vec.reshape(1, 1, -1), axis=-1)
                coeff = (avg_proj - current_projections)[..., None]
                residuals = residuals + coeff * unit_vec.reshape(1, 1, -1)
            else:
                raise ValueError(f"Unknown steering method: {self.steering_method}")

            return residuals

        self._hooks = []
        target_layer_name = f"model.layers.{self.optimal_layer}"
        for name, module in model.named_modules():
            if name == target_layer_name:
                hook = module.register_forward_hook(
                    lambda m, i, o: steering_hook(i, o)
                )
                self._hooks.append(hook)
                break

        print(f"Installed {len(self._hooks)} hooks on layers: {target_layer_name}")

    def _parse_steering_config(self) -> Dict:
        config_path = os.environ.get("VLLM_ACTSTEER_CONFIG")
        with open(config_path, "r") as f:
            config = json.load(f)

        steering_config = config.get("steering", {})
        return {
            "method": steering_config.get("method", "adjust_rs"),
            "optimal_layer": int(steering_config.get("optimal_layer", 15)),
            "coefficient": float(steering_config.get("coefficient", 0)),
            "vector_path": steering_config.get("vector_path"),
            "apply_at_all_positions": steering_config.get("apply_at_all_positions", True),
        }

    def execute_model(self, *args, **kwargs):
        return super().execute_model(*args, **kwargs)
