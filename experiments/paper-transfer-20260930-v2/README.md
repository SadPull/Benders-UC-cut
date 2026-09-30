# SA-Benders paper transfer experiment

Release: v0.12.0 (2026-10-01). Experiment: 2026-09-30.

- [Chinese report](report.md): six methods, five seeds each; all 30 runs physically validated.
- [Summary](summary.csv), [paired comparisons](paired.csv), [time checkpoints](checkpoints.csv), and [per-run metrics](runs.csv).
- [Validation audit](audit.json) and [independent reference](reference.json).
- [Complete reproducibility archive](reproducibility.zip), verified by [SHA256SUMS](SHA256SUMS).

The archive contains the complete original experiment directory, including all 30 run JSON files, detailed trajectories, input/configuration manifest, source snapshot, working-tree provenance, validation tests, and report generator. No LLM requests or decommitment candidates were generated in this study. Prevalidation and interrupted pilot data are not included.

The exact experiment source is in `paper-transfer-20260930-v2/source/` inside the archive. Its manifest was recorded before release documentation was updated. Use that snapshot, its dependencies and the recorded single-thread environment when reproducing the solver fingerprint. Reports can be regenerated with the archived `report-source.py`. Git tracks the compressed archive to avoid adding 133 MB of expanded trace data.

To restore locally, verify the archive checksum and extract it under `results/`. Existing results should be preserved rather than overwritten. Ordinary new runs use `python -m suc_benders.paper_experiment` from the repository root; choose a fresh `--output-dir` for a new experiment.
