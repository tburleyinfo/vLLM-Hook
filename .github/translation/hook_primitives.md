# General hook correspondence

A source framework's forward-hook registration can map to callable-module
replacement on a target framework. Retain the original module, install a wrapper
at the owning object's attribute (for example through attribute assignment or
`setattr`), and restore that original reference when instrumentation ends.

A wrapper may observe arguments and results while delegating the original call.
The translation must preserve outputs, parameters, mutable state, call arguments,
module discovery and cleanup behavior. Attribute replacement alone does not prove
these properties or expose intermediate values computed inside the original call.

This is a possible platform correspondence, not a completed port. It is grounded
in attribute replacement and callable composition present in the supplied target
library source. Determine which instances satisfy it by inspecting that source.
No application-specific class, hook location, tensor reconstruction, metadata
schema, request mapping, or implementation helper is supplied by this note.


# General array-conversion correspondence

MLX bfloat16 arrays cannot be converted directly to NumPy in this environment.
Cast to a NumPy-compatible dtype inside MLX before crossing the array boundary:

```python
np.asarray(tensor.astype(mx.float32))
```

`np.asarray(tensor).astype(np.float32)` fails before the NumPy cast can run.
A standalone test reproduced the buffer-format error for bfloat16 and confirmed
that the MLX-first cast works for bfloat16, float16 and float32.
This is an array interoperability rule. Casting an exported array does not change
the model's actual computation precision or establish behavioral parity.
