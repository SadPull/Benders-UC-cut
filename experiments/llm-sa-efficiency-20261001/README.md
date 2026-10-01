# LLM / SA efficiency — 2026-10-01

See [中文结论](findings.md), [detailed report](report.md), and [audit](audit.json).

The implemented dynamic whole-horizon study compares SA with SA+LLM at the
same 600-second end-to-end budget. Four seed pairs completed; both seed-7
traces were interrupted and retained without rerunning. No usable LLM
candidates were returned by the selected deployment: all completed responses
truncated at 32000 reported reasoning tokens despite requesting 65536.
This result evaluates that configuration, not the potential of valid advice.

`reproducibility.zip` includes the exact run source snapshot, input, software
versions, raw event streams, redacted request/response bodies, every validated
schedule, reference objective and reference-source manifest, passing test logs,
plot and audit scripts. Secrets are excluded. `SHA256SUMS` authenticates these
published files. The archive was verified against the local study byte for byte.
The optimization implementation was committed as `c19c6ed` (integration:
`e5922cb`); the snapshot hashes in `manifest.json` identify the actual files.

## Reproduce reports without optimization

From the repository root, extract the archive into
`results/llm-sa-efficiency-20261001/`, then run:

```powershell
python -m suc_benders.efficiency_experiment --output-dir results/llm-sa-efficiency-20261001 --report-only
python -c "import runpy; runpy.run_path('results/llm-sa-efficiency-20261001/audit_experiment.py', run_name='__main__')"
python -c "import runpy; runpy.run_path('results/llm-sa-efficiency-20261001/plot_results.py', run_name='__main__')"
```

The audit performs a local secret check using `LLM_API_KEY` from the environment
or ignored `.env`; it makes no network or optimization calls. Report generation
needs no API key. The saved source can be used when exact-byte source
fingerprints differ from a checkout due to line-ending conversion.

## Start a separate fresh study

The default entry point verifies the existing reference at
`results/paper-transfer-20260930-v2/`. If it is absent in a fresh checkout, restore
`reference.json` and `reference-source-manifest.json` from this archive as that
directory's `reference.json` and `manifest.json`, respectively. These files were
copied from the original study, not regenerated. Set `LLM_API_KEY` locally and
run `python -m suc_benders.efficiency_experiment` for a NEW timestamped output.
The endpoint/model/reasoning settings remain those requested for this study.

`--resume` requires matching source, input, settings and software fingerprints;
it skips all existing runs, including interrupted runs. It does not replenish
budgets or silently rerun an interrupted trial. The completed-state resume was
verified to leave all ten event-stream hashes unchanged.
