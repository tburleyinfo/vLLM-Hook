"""MIA capture schema; head selection and hook mode belong outside config."""
import math

def capture_config(text_config, dimensions):
    scale = 1.0 / math.sqrt(dimensions["head_dim"])
    if not math.isclose(dimensions["attention_multiplier"], scale, rel_tol=1e-12, abs_tol=0.0):
        raise RuntimeError("Live attention uses a nonstandard scale; refuse normalization")
    return {"num_attention_heads": dimensions["num_attention_heads"],
            "num_key_value_heads": dimensions["num_key_value_heads"],
            "hidden_size": int(text_config.hidden_size), "head_dim": dimensions["head_dim"],
            "attention_multiplier": scale}
