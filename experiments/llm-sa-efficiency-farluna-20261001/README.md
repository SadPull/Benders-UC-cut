# Farluna LLM / SA efficiency rerun

See [中文结论](findings.md), [detailed report](report.md), [LLM attribution](llm-attribution.json), and [audit](audit.json).

This is a separate rerun of the v0.14.0 protocol using
`https://api.farluna.tech/v1` and `farluna/deepseek-v4.1-flash`, with
`reasoning_effort=max` and `max_tokens=65536`. Each run retains the same
600-second end-to-end budget. The request timeout is 300 seconds, capped by
the remaining run budget; v0.14.0 used 180 seconds. Historical results are
preserved. This is a comparison of the two strategies within the new study,
not an isolated estimate of the effect of changing providers.

`reproducibility.zip` contains the exact source snapshot, input, configuration,
software versions, raw events, redacted requests and responses, complete
validated schedules, reference provenance, regression log, single-request
provider check, and post-run scripts. `SHA256SUMS` lists hashes of published
files. The experiment implementation was committed as `1667645`; source
hashes in `manifest.json` identify the exact bytes used during the study.

## Reproduce reports without optimization

Extract the archive into `results/llm-sa-efficiency-farluna-20261001/` from
the repository root, then run:

```powershell
python -m suc_benders.efficiency_experiment --output-dir results/llm-sa-efficiency-farluna-20261001 --report-only
python -c "import runpy; runpy.run_path('results/llm-sa-efficiency-farluna-20261001/audit_experiment.py', run_name='__main__')"
python -c "import runpy; runpy.run_path('results/llm-sa-efficiency-farluna-20261001/analyze_llm.py', run_name='__main__')"
python -c "import runpy; runpy.run_path('results/llm-sa-efficiency-farluna-20261001/plot_results.py', run_name='__main__')"
```

The audit loads `LLM_API_KEY` locally only to check that it is absent from
artifacts; it makes no API or optimization calls. Report and plot generation
need no key. Use the source snapshot when checkout line endings differ from
recorded source fingerprints. The saved audit also records the separately
verified completed-state resume; rerunning the audit regenerates base checks.

## Start a separate study

The experiment expects the same input reference under
`results/paper-transfer-20260930-v2/`. In a fresh checkout, restore this
archive's `reference.json` there and `reference-source-manifest.json` as
`manifest.json`. Set credentials in the environment or ignored `.env`.
Use a new output directory for a new experiment.

`--resume` requires identical source/input/configuration/software fingerprints,
skips existing runs including interrupted trials, and does not reset budgets.
The reference is used only in reporting, never as advice or a stopping rule.
