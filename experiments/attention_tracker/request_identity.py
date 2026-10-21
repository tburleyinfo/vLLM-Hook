"""Snapshot explicit vLLM request identity before output cleanup removes it."""
from contextlib import contextmanager
import json
from pathlib import Path

@contextmanager
def capture_request_mapping(engine, artifact_path="request-id-map.json"):
    processor = engine.output_processor
    original = processor.add_request
    mapping = {}
    def record(request, *args, **kwargs):
        external = request.external_req_id
        internal = request.request_id
        if external is None:
            raise RuntimeError("vLLM did not assign an external request identity")
        result = original(request, *args, **kwargs)
        mapping.setdefault(str(external), []).append(str(internal))
        print(f"[identity] external={external} internal={internal}", flush=True)
        return result
    processor.add_request = record
    try:
        yield mapping
    finally:
        processor.add_request = original
        Path(artifact_path).write_text(json.dumps(mapping, indent=2) + "\n")

def internal_id(mapping, external):
    ids = mapping.get(str(external), [])
    if len(ids) != 1:
        raise RuntimeError(f"Expected one internal request for {external!r}, got {ids!r}")
    return ids[0]
