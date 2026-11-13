# Run validation manually

Use Python 3.12 and run from the root of a checkout of this PR. The clone must contain the pinned source commit `8c0f2f4df88511e6e487d6af813dc7d8c2b91027`. The runner verifies the attached patch and profile and reconstructs the candidate from that commit. Use a new output directory for each execution.

```bash
KIT="translation-experiments/graph-48a4ad62cb7d07cecb389d0dec5115d19d81fbdb7c6d1a6774a7a2bdc892f2a2/validation"
PROFILE="$KIT/profile/profile.json"
RESULTS="$(mktemp -d)"
```

On Linux, run the CPU reference. On Apple Silicon macOS, run the Metal candidate. Set the variables above on each machine.

```bash
python3.12 "$KIT/hosted_runtime.py" --bundle "$KIT" --profile "$PROFILE" --backend cpu --output "$RESULTS/cpu"
python3.12 "$KIT/hosted_runtime.py" --bundle "$KIT" --profile "$PROFILE" --backend target --output "$RESULTS/target"
```

Copy the CPU and Metal result folders to the same machine and compare them. If you already have a CPU report for the exact same patch, source and profile, you can reuse it.

```bash
python3.12 "$KIT/hosted_runtime.py" --bundle "$KIT" --profile "$PROFILE" --cpu "$RESULTS/cpu/result.json" --target "$RESULTS/target/result.json" --output "$RESULTS/comparison.json"
```

Keep both backend reports and `comparison.json` with the experiment record.
