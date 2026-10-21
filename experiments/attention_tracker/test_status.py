"""Verify safe operational progress from partial Copilot event logs."""
import json
from pathlib import Path
import tempfile
import unittest
import status


class StatusTests(unittest.TestCase):
    def test_activity_uses_tools_not_reasoning_or_payloads(self):
        events = [
            {"type": "assistant.reasoning", "data": {"content": "private reasoning"}},
            {"type": "model.call_start", "data": {"model": "example-model"}},
            {"type": "tool.execution_start", "data": {"toolCallId": "1", "toolName": "apply_patch", "arguments": "private patch"}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "1", "success": True, "fileEdits": [{"path": "/candidate/port.py"}]}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "translation.log"
            log.write_text("\n".join(json.dumps(e) for e in events) + '\n{"partial":')
            result = status.translation_activity(log)
            self.assertIn("1 tool calls finished", result)
            self.assertIn("port.py", result)
            self.assertIn("example-model", result)
            self.assertNotIn("private", result)


if __name__ == "__main__":
    unittest.main()
