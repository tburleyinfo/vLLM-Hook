# Run validation manually

Use Python 3.12 and run from the root of a checkout of this PR. The clone must contain the pinned source commit `8c0f2f4df88511e6e487d6af813dc7d8c2b91027`. The runner verifies the attached patch and profile and reconstructs the candidate from that commit. Use a new output directory for each execution.

```bash
KIT="translation-experiments/graph-6a70a05a5aa071797e0d9f8abb0edf48daa6ac54779e5948019efee684006301/validation"
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
