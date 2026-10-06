"""Verify the completed batch without new sampling or changing archived files."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "snapshot"))

from suc_benders.block_master_experiment import native_checks, relative_gap
from suc_benders.block_master_model import fixed_master_value, priced_blocks
from suc_benders.data import load_suc_data
from suc_benders.paper_model import Cut, IndexedUC, grid_bounds, name


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def point(record):
    return {tuple(k): v for k, v in record}


def check_close(actual, expected):
    if actual is None or expected is None:
        assert actual is expected, (actual, expected)
    else:
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-6), (actual, expected)


checksums = [line.split("  ", 1) for line in (ROOT / "SHA256SUMS.txt").read_text().splitlines()]
for expected, relative in checksums:
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
manifest, summary = read(ROOT / "manifest.json"), read(ROOT / "summary.json")
for relative, expected in manifest["source_sha256"].items():
    assert hashlib.sha256((ROOT / "snapshot" / relative).read_bytes()).hexdigest() == expected, relative
assert summary["complete"] and read(ROOT / "state.json")["status"] == "completed"
assert not any(summary[k] for k in ("uncertified_snapshots", "incomplete_traces", "incomplete_snapshots"))
elapsed = read(ROOT / "source-verification.json")["elapsed_seconds"]
assert elapsed < manifest["budget_seconds"]
data, scenarios = load_suc_data(ROOT / "snapshot/data/rts24/units.json", ROOT / "snapshot/data/rts24/stochastic.json")
indexed = IndexedUC(data, scenarios)
prices = read(ROOT / "pricing.json")["multipliers"]
blocks = {}
rows = []
csv_samples = list(csv.DictReader((ROOT / "samples.csv").open(encoding="utf-8", newline="")))
csv_samples = {(r["snapshot"], int(r["seed"]), int(r["sample"])): r for r in csv_samples}
snapshot_csv = {r["snapshot"]: r for r in csv.DictReader((ROOT / "snapshot-summary.csv").open(encoding="utf-8", newline=""))}
slots = set()
reference_residuals = []
for definition in manifest["traces"]:
    trace = read(ROOT / "traces" / (definition["id"] + ".json"))
    assert trace["status"] in {"converged", "five_iterations"}
    assert set(trace["slots"]) == {"initial", "final"}
    slots.update(trace["slots"].values())

for folder in sorted((ROOT / "snapshots").iterdir()):
    meta, encoding = read(folder / "snapshot.json"), read(folder / "encoding.json")
    samples, reference = read(folder / "samples.json"), read(folder / "reference.json")
    spec = meta["spec"]
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert identity == meta["identity"] and read(folder / "encoding-audit.json")["passed"]
    assert samples["complete"] and not samples["pending_seeds"]
    assert encoding["completed_seeds"] == manifest["seeds"]
    assert [c["seed"] for c in samples["calls"]] == manifest["seeds"]
    assert all(c["returned"] == 16 and not c["late"] for c in samples["calls"])
    key = (meta["hours"], meta["price"])
    if key not in blocks:
        blocks[key] = priced_blocks(indexed, key[0], [0.] * len(prices) if key[1] == "zero" else prices)[0]
    block = blocks[key][meta["block"]]
    cuts = [Cut(point(c["coefficients"]), c["constant"], c["feasible"]) for c in spec["cuts"]]
    assert block.times == tuple(spec["times"])
    check_close(block.constant, spec["objective_constant"])
    assert reference["certified"] and reference["status"] == 2
    reference_native = native_checks(block, cuts, point(reference["point"]), reference["theta"])
    assert reference_native["discrete"]["feasible"] and reference_native["feasibility_cut_violation"] <= 1e-6
    # Audit classical theta at 1e-5; keep SA's original 1e-6 checks unchanged.
    # Explicitly flag the one reference that exceeds the strict SA tolerance.
    assert reference_native["optimality_violation"] <= 1e-5
    if reference_native["optimality_violation"] > 1e-6:
        reference_residuals.append(dict(snapshot=folder.name, optimality_violation=reference_native["optimality_violation"]))
    assert abs(fixed_master_value(block, cuts, point(reference["point"])) - reference["objective"]) <= 1e-5
    assert -1e-6 <= reference["objective"] - reference["lower_bound"] <= manifest["mip_gap"] * max(1., abs(reference["objective"])) + 1e-6
    with np.load(folder / "qubo.npz") as qubo:
        assert list(qubo["variable_names"]) == encoding["variables"]
        assert qubo["matrix"].shape == (encoding["bits"], encoding["bits"])
    lower, upper = grid_bounds(spec["lower"], spec["upper"], spec["theta_step"])
    steps = int(round((upper - lower) / spec["theta_step"]))
    n = int(math.ceil(math.log2(steps + 1))) if steps else 0
    weights = [1 << j for j in range(n - 1)] + [steps - ((1 << (n - 1)) - 1)] if n else []
    by_seed = {seed: [s for s in samples["samples"] if s["seed"] == seed] for seed in manifest["seeds"]}
    for seed, records in by_seed.items():
        with np.load(folder / f"samples-{seed}.npz") as raw:
            bits_list = raw["samples"]
        assert len(records) == len(bits_list) == 16
        for sample, bits in zip(records, bits_list):
            assert sample["sample"] < 16 and set(map(int, bits)) <= {0, 1}
            values = dict(zip(encoding["variables"], map(int, bits)))
            decoded = {k: int(block.fixed[k]) if k in block.fixed else values.get(name(k), 0) for k in block.bkeys}
            assert decoded == point(sample["raw_point"])
            raw_id = hashlib.sha256(bytes(int(decoded[k]) for k in block.bkeys)).hexdigest()
            assert raw_id == sample["raw_id"]
            objective = lower + spec["theta_step"] * sum(w * values.get(f"z_{i}", 0) for i, w in enumerate(weights))
            theta = objective - sum(block.costs[k] * decoded[k] for k in block.bkeys)
            check_close(theta, sample["decoded_theta"])
            check_close(objective + block.constant, sample["decoded_objective"])
            native = native_checks(block, cuts, decoded, theta)
            assert native == sample["native_checks"]
            residuals = {r["label"]: r["constant"] + sum(v * values[k] for k, v in r["coefficients"].items()) for r in encoding["inequalities"]}
            assert {k: v for k, v in residuals.items() if v} == sample["encoded_checks"]["nonzero_integer_residuals"]
            and_ok = all(values[v] == decoded[("v", int(v.split("_")[1]), int(v.split("_")[2]), -1)] * decoded[("w", int(v.split("_")[1]), int(v.split("_")[2]) + 1, -1)] for v in values if v.startswith("and_"))
            encoded_ok = block.discrete_feasible(decoded) and and_ok and not any(residuals.values())
            assert encoded_ok == sample["raw_encoded_feasible"]
            assert (native["feasible"] and encoded_ok) == sample["full_master_feasible"]
            value = fixed_master_value(block, cuts, decoded)
            check_close(value, sample["classical_fixed_raw_value"])
            check_close(relative_gap(value, reference), sample["classical_fixed_raw_gap"])
            check_close(relative_gap(objective + block.constant, reference) if sample["full_master_feasible"] else None, sample["raw_gap"])
            repaired = point(sample["repaired_point"])
            changes = sum(decoded[k] != repaired[k] for k in block.bkeys if k[0] == "u")
            assert changes == sample["repair_bits"]
            check_close(changes / (data.n_units * len(block.times)), sample["repair_fraction"])
            check_close(fixed_master_value(block, cuts, repaired), sample["repaired_value"])
            check_close(relative_gap(sample["repaired_value"], reference), sample["repaired_gap"])
            csv_row = csv_samples[(folder.name, seed, sample["sample"])]
            for field in ("raw_discrete_feasible", "full_master_feasible"):
                assert csv_row[field] == str(sample[field])
            for field in ("raw_gap", "classical_fixed_raw_gap", "repaired_gap", "repair_fraction"):
                check_close(float(csv_row[field]) if csv_row[field] else None, sample[field])
    assert int(snapshot_csv[folder.name]["bits"]) == encoding["bits"]
    rows.append((meta, samples, encoding))
    print(f"verified {folder.name}", flush=True)

assert slots == {m["id"] for m, _, _ in rows}
assert len(rows) == summary["snapshots"] == 50
assert sum(len(r["calls"]) for _, r, _ in rows) == summary["sa_calls"] == 150
assert sum(len(r["samples"]) for _, r, _ in rows) == len(csv_samples) == summary["candidate_count"] == 2400
groups = {}
for hours in (24, 3, 6):
    own = [(m, s, e) for m, s, e in rows if m["hours"] == hours]
    ss = [s for _, r, _ in own for s in r["samples"]]
    saved = summary["groups"][str(hours)]
    duplicate_count = sum(len(r["samples"]) - len({s["repaired_id"] for s in r["samples"]}) for _, r, _ in own)
    check_close(sum(s["raw_discrete_feasible"] for s in ss) / len(ss), saved["raw_discrete_feasible_rate"])
    check_close(sum(s["repair_fraction"] for s in ss) / len(ss), saved["repair_fraction"])
    check_close(duplicate_count / len(ss), saved["repaired_duplicate_rate"])
    assert sum(s["full_master_feasible"] for s in ss) == saved["full_master_feasible"]
    assert sum(any(s["classical_fixed_raw_gap"] is not None and s["classical_fixed_raw_gap"] <= .05 for s in r["samples"]) for _, r, _ in own) == saved["certified_with_5pct_unmodified_commitment"]
    gaps = [s["classical_fixed_raw_gap"] for s in ss if s["classical_fixed_raw_gap"] is not None]
    groups[str(hours)] = dict(bits_range=[min(e["bits"] for _, _, e in own), max(e["bits"] for _, _, e in own)],
        normalized_objective_range=[min(e["normalized_energy_unit"] for _, _, e in own), max(e["normalized_energy_unit"] for _, _, e in own)],
        raw_native_master_feasible=sum(s["raw_native_master_feasible"] for s in ss),
        raw_encoded_feasible=sum(s["raw_encoded_feasible"] for s in ss),
        raw_feasible_unique=sum(len({s["raw_id"] for s in r["samples"] if s["raw_discrete_feasible"]}) for _, r, _ in own),
        native_violation_samples=dict(Counter(label for s in ss for label in s["native_checks"]["discrete"]["violations"])),
        capacity_violation_samples=sum(s["native_checks"]["discrete"]["capacity_violation_mw"] > 1e-6 for s in ss),
        fixed_raw_eligible_snapshots=sum(any(s["classical_fixed_raw_gap"] is not None for s in r["samples"]) for _, r, _ in own),
        best_fixed_raw_gap=min(gaps) if gaps else None,
        best_repaired_gap=min(s["repaired_gap"] for s in ss if s["repaired_gap"] is not None))
result = dict(passed=True, original_files_verified=len(checksums), frozen_source_files_verified=len(manifest["source_sha256"]),
    snapshots_verified=len(rows), traces_verified=len(manifest["traces"]), sa_calls_verified=150, samples_verified=len(csv_samples),
    elapsed_seconds=elapsed, new_sa_calls=0, original_archive_modified=False,
    reference_theta_audit_tolerance=1e-5, sa_native_tolerance_unchanged=1e-6,
    references_exceeding_strict_native_theta_tolerance=reference_residuals,
    maximum_certified_reference_mip_gap=max(read(ROOT / "snapshots" / m["id"] / "reference.json")["mip_gap"] for m, _, _ in rows),
    groups=groups)
(ROOT / "review/audit.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
