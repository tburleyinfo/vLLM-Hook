"""Read-only status display; never starts or stops an experiment."""
import argparse
from datetime import datetime, timezone
import json
import os
import shutil
import shlex
from pathlib import Path
import time


def translation_activity(log):
    """Display operational events only, excluding internal reasoning and payloads."""
    events = []
    with log.open() as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # A writer may be in the middle of its latest event.
            events.append(event)
    if not events:
        return "Copilot starting; no activity events yet"
    starts, completed, edits = {}, set(), set()
    model = None
    latest = "Copilot starting"
    first_timestamp = None
    for event in events:
        data = event.get("data", {})
        timestamp = event.get("timestamp")
        if first_timestamp is None and timestamp:
            first_timestamp = timestamp
        kind = event.get("type")
        if kind == "model.call_start":
            model = data.get("model", model)
            latest = "Waiting for Copilot's next response"
        elif kind == "tool.execution_start":
            tool = data.get("toolName", "tool")
            starts[data.get("toolCallId")] = tool
            latest = {"view": "Reading source", "grep": "Searching source", "glob": "Finding files",
                      "apply_patch": "Writing candidate changes", "create": "Creating a file",
                      "edit": "Editing a file"}.get(tool, "Using " + tool)
            arguments = data.get("arguments", {})
            if isinstance(arguments, dict):
                path = arguments.get("path") or arguments.get("file_path")
                if isinstance(path, str):
                    latest += ": " + Path(path).name
        elif kind == "tool.execution_complete":
            completed.add(data.get("toolCallId"))
            for edit in data.get("fileEdits", []):
                if edit.get("path"):
                    edits.add(Path(edit["path"]).name)
            if data.get("success") is False:
                latest = "Latest tool reported a failure; Copilot may retry"
            else:
                latest = "Tool finished; waiting for Copilot's next step"
    elapsed = ""
    if first_timestamp:
        try:
            seconds = int((datetime.now(timezone.utc) - datetime.fromisoformat(first_timestamp.replace("Z", "+00:00"))).total_seconds())
            elapsed = f"{seconds // 60}m {seconds % 60}s elapsed; "
        except ValueError:
            pass
    detail = f"{elapsed}{len(completed)} tool calls finished"
    if model:
        detail += "; model: " + model
    if edits:
        detail += "\n  Files changed: " + ", ".join(sorted(edits))
    return latest + "\n  " + detail


def read_result(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def overview(run):
    """Explain waiting states without claiming a process is alive from an old log."""
    manifest = read_result(run / "manifest.json") or {}
    conditions = ("source_only", "source_validated_dag")
    started = any((run / name / filename).exists()
                  for name in conditions for filename in ("translation.log", "translation-result.json"))
    lines = ["Read-only monitor: this command does not launch jobs."]
    if not started:
        lines.append("WAITING TO START: neither translation has been launched. No translation timer is running.")
        pinned = Path(__file__).resolve().parent / ".tools/copilot-1.0.97/node_modules/.bin/copilot"
        executable = os.environ.get("ATTN_COPILOT_BIN") or (str(pinned) if pinned.is_file() else "copilot")
        found = shutil.which(executable)
        if not found:
            lines.append("BLOCKER: Copilot CLI not found: " + executable)
            lines.append("Next: restore the pinned Copilot CLI and set ATTN_COPILOT_BIN to its executable.")
        else:
            lines.append("Copilot executable found: " + found + " (version and login not checked)")
        if (read_result(run / "freeze-result.json") or {}).get("status") != "pass":
            lines.append("BLOCKER: shared inputs are not frozen successfully yet.")
        pilot = manifest.get("translator", {}).get("model") == "auto"
        if pilot:
            lines.append("Model setting: Auto; translation must be labeled a pilot.")
        command = "python3 experiments/attention_tracker/run.py --stage translate --run-dir " + shlex.quote(str(run))
        if pilot:
            command += " --pilot"
        lines.append("Launch separately from the repository root, after resolving blockers:\n  " + command)
    else:
        pending = [name for name in conditions if not (run / name / "translation-result.json").exists()]
        if pending:
            lines.append("Translation completion pending: " + ", ".join(pending) + ". Log activity alone does not prove a process is alive.")
        else:
            lines.append("Both translation attempts have completion reports. Check their pass/fail states below.")
        if not any((run / name / "metal-result.json").exists() for name in conditions):
            lines.append("Metal candidate evaluation has not started; translation does not launch evaluation automatically.")
    return "\n".join(lines)


def describe(run, result_name, log_name, verbose=False):
    result = run / result_name
    data = read_result(result)
    if data is not None:
        description = data.get("status", "unknown") + (": " + data["error"] if data.get("error") else "")
        if result_name == "live-worker-result.json" and data.get("status") == "pass":
            if data.get("precision_verified"):
                description += "\n  Actual model precision: float32 verified"
            if data.get("request_identity_verified"):
                description += "\n  Worker/output request IDs: explicit mapping verified"
        if verbose:
            description += "\n  Result saved: " + datetime.fromtimestamp(result.stat().st_mtime).astimezone().isoformat(timespec="seconds")
            timing = read_result(result.with_name(result.name.replace("-result.json", "-process.json"))) or {}
            details = {**timing, **data}
            for key in ("stage", "seconds", "exit_code", "symbols", "relationships"):
                if key in details:
                    description += f"\n  {key}: {details[key]}"
            description += "\n  Report: " + str(result)
        return description
    log = run / log_name
    if not log.exists():
        return "not started" if not result.exists() else "result is being written; retrying next refresh"
    age = max(0, int(time.time() - log.stat().st_mtime))
    description = f"no completion report; last log update {age}s ago (process state unknown)"
    if age >= 60:
        description += "\n  Quiet log: the job may be computing, waiting, interrupted, or stopped."
    if log.name == "translation.log":
        description += "\n  " + translation_activity(log)
    if verbose:
        description += f"\n  Log: {log} ({log.stat().st_size:,} bytes)"
    return description


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show result times, timings, report paths and log sizes")
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        parser.error("Run directory does not exist")
    args.run_dir = args.run_dir.resolve()
    try:
        while True:
            print(f"\nStatus at {datetime.now().strftime('%H:%M:%S')}", flush=True)
            print(overview(args.run_dir), flush=True)
            if args.verbose:
                print("Run: " + str(args.run_dir), flush=True)
            for label, result, log in (
                ("Platform discovery", "discovery-result.json", "discovery.log"),
                ("Library graph retrieval", "target-retrieval-result.json", "retrieval.log"),
                ("Platform witnesses", "platform-probe-result.json", "platform-probe.log"),
                ("Live worker observation", "live-worker-result.json", "live-worker.log"),
                ("Shared evidence freeze", "freeze-result.json", "freeze.log"),
                ("CPU baseline", "cpu-result.json", "cpu.log"),
                ("Source compatibility check (optional diagnostic)", "metal-preflight-result.json", "metal-preflight.log"),
                ("Shared-context translation", "source_only/translation-result.json", "source_only/translation.log"),
                ("Graph translation", "source_validated_dag/translation-result.json", "source_validated_dag/translation.log"),
                ("Shared-context Metal test", "source_only/metal-result.json", "source_only/metal.log"),
                ("Graph Metal test", "source_validated_dag/metal-result.json", "source_validated_dag/metal.log"),
            ):
                print(f"{label}: {describe(args.run_dir, result, log, args.verbose)}", flush=True)
            if not args.watch:
                return
            time.sleep(5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
