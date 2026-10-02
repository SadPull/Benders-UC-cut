"""Rebuild the RTS-24 derived inputs from pinned, local public source files."""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "case24_ieee_rts.m": {
        "url": "https://raw.githubusercontent.com/MATPOWER/matpower/8b326f00d496864ed81169f7e684568fc1f7006d/data/case24_ieee_rts.m",
        "commit": "8b326f00d496864ed81169f7e684568fc1f7006d",
        "sha256": "a383a9001fd03ab54b2bc590364a71119fd07e82ab814adfd824ce08f9eb26ce",
    },
    "RTS79_ucacopf.dat": {
        "url": "https://raw.githubusercontent.com/anyacastillo/ucacopf/001cf29dc2e9b3d26e71146f82db35fa379b4dca/RTS79_ucacopf.dat",
        "commit": "001cf29dc2e9b3d26e71146f82db35fa379b4dca",
        "sha256": "ed1aad8c7a2736edfa411d7a3e45c550e8e5a9e5879224db4e18c529253446eb",
    },
}


def matrix(text, name):
    body = re.search(r"mpc\." + name + r"\s*=\s*\[(.*?)\];", text, re.S).group(1)
    body = re.sub(r"%[^\n]*", "", body)
    return [[Decimal(v) for v in row.split()] for row in body.split(";") if row.strip()]


def table(text, header):
    body = next(part for part in text.split(";") if header in part)
    return [line.split() for line in body.split(":=", 1)[1].strip().splitlines()]


def convert(source_dir=ROOT / "source"):
    source = {}
    for name, info in SOURCES.items():
        raw = (source_dir / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != info["sha256"]:
            raise ValueError(f"source fingerprint mismatch: {name}")
        source[name] = raw.decode("utf-8")
    mp, uc = source["case24_ieee_rts.m"], source["RTS79_ucacopf.dat"]
    generators, costs = matrix(mp, "gen"), matrix(mp, "gencost")
    records = table(uc, "PowerGeneratedT0")
    ids = [int(row[0]) for row in records]
    expected = [i + 1 for i, row in enumerate(generators) if row[8] > 0]
    if ids != expected or len(ids) != 32:
        raise ValueError("UC and MATPOWER generator IDs do not match")
    base = Decimal(re.search(r"param baseMVA := (.*?);", uc).group(1))
    horizon = int(re.search(r"param NumTimePeriods := (.*?);", uc).group(1))
    declared = [int(v) for v in re.search(r"set TimePeriod := (.*?);", uc).group(1).split()]
    demands = {(int(b), int(t)): Decimal(v) for b, t, v in table(uc, "TimePeriod Demand")}
    buses = [int(v) for v in re.search(r"set Bus := (.*?);", uc).group(1).split()]
    if len(demands) != len(buses) * horizon or set(demands) != {(b, t) for b in buses for t in range(1, horizon + 1)}:
        raise ValueError("incomplete hourly bus demand")
    if horizon != 24 or sorted(set(range(1, 25)) - set(declared)) != [15]:
        raise ValueError("unexpected source time-period declarations")
    load = [float(base * sum(demands[b, t] for b in buses)) for t in range(1, horizon + 1)]
    data = {"n_units": len(ids), "horizon": horizon, "base_load": load, "reserve_fraction": 0.0}
    fields = ("pmax", "pmin", "pstart", "pshut", "min_up", "min_down", "ramp_up", "ramp_down",
              "initial_duration", "initial_output", "cost_a", "cost_b", "cost_c", "startup_cost", "shutdown_cost")
    data.update({key: [] for key in fields})
    mapping = []
    for row in records:
        ident = int(row[0])
        g, c = generators[ident - 1], costs[ident - 1]
        values = [Decimal(v) for v in row]
        if (g[9], g[8]) != (values[3] * base, values[4] * base):
            raise ValueError(f"capacity mismatch for generator {ident}")
        if c[0] != 2 or c[3] != 3:
            raise ValueError("expected quadratic MATPOWER cost")
        duration, up, down = values[2], values[5], values[6]
        if any(v != int(v) for v in (duration, up, down)):
            raise ValueError("expected integer hourly durations")
        item = dict(pmax=g[8], pmin=g[9], pstart=min(g[8], base * values[9]),
                    pshut=min(g[8], base * values[10]), min_up=int(up), min_down=int(down),
                    ramp_up=base * values[7], ramp_down=base * values[8],
                    initial_duration=int(duration), initial_output=base * values[1],
                    cost_a=c[4], cost_b=c[5], cost_c=c[6], startup_cost=c[1], shutdown_cost=c[2])
        for key, value in item.items():
            data[key].append(float(value) if isinstance(value, Decimal) else value)
        uc_bus = next(int(m.group(1)) for m in re.finditer(r"set ThermalGeneratorsAtBus\[(\d+)\] :=([^;]+);", uc)
                      if ident in [int(v) for v in m.group(2).split()])
        if uc_bus != int(g[0]) + 100:
            raise ValueError(f"bus mapping mismatch for generator {ident}")
        mapping.append({"index": len(mapping), "source_generator_id": ident, "matpower_bus": int(g[0]),
                        "uc_bus": uc_bus, "raw_startup_limit_MW": float(base * values[9]),
                        "raw_shutdown_limit_MW": float(base * values[10])})
    if sum(data["pmax"]) != 3405 or max(load) != 2850:
        raise ValueError("unexpected RTS capacity or peak demand")
    provenance = {
        "name": "IEEE RTS-24 derived copperplate UC/SUC (fixed startup costs)",
        "sources": SOURCES, "generator_mapping": mapping,
        "field_sources": {
            "pmin,pmax": "MATPOWER gen columns 10,9 (MW); cross-checked against UC data times baseMVA",
            "cost_a,cost_b,cost_c": "MATPOWER gencost columns 5,6,7; evaluated on MW, no scaling",
            "startup_cost,shutdown_cost": "MATPOWER gencost columns 2,3; fixed dollars per event",
            "min_up,min_down": "UC MinimumUpTime, MinimumDownTime; hours",
            "ramp_up,ramp_down": "UC NominalRampUpLimit, NominalRampDownLimit times baseMVA; MW per hourly step",
            "pstart,pshut": "min(Pmax, UC StartupCapacityLimit/ShutdownCapacity times baseMVA)",
            "initial_duration": "UC UnitOnT0State; signed hours",
            "initial_output": "UC PowerGeneratedT0 times baseMVA; MW",
            "base_load": "sum UC Demand over all 24 buses times baseMVA; Decimal arithmetic",
            "reserve_fraction": "UC ReserveRequirement is zero; stochastic profile overrides to project assumption 0.05",
        },
        "source_corrections": ["TimePeriod omits hour 15; use NumTimePeriods=24 and the complete 24-by-24 Demand table.",
                               "Exclude generator 15 (synchronous condenser) despite its presence in the UC ThermalGenerator declaration."],
        "model_choices": ["Aggregate bus loads; omit network, reactive power and contingencies.",
                          "Use MATPOWER fixed startup costs instead of UC StartupCosts/StartupLags.",
                          "Six U50 hydro units are dispatchable units without water or energy budgets.",
                          "Retain project ramp and terminal rules; this is not a reproduction of UC+ACOPF."],
    }
    deterministic = {"reserve_fraction": 0.0, "scenarios": [{"name": "base", "probability": 1.0, "load_factor": 1.0}]}
    stochastic = {"reserve_fraction": 0.05, "scenarios": [
        {"name": name, "probability": prob, "load_factor": factor}
        for name, prob, factor in (("low", .25, .9), ("base", .5, 1.), ("high", .25, 1.05))]}
    return {"units.json": data, "deterministic.json": deterministic,
            "stochastic.json": stochastic, "provenance.json": provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify generated files without writing")
    args = parser.parse_args()
    for name, value in convert().items():
        content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        path = ROOT / name
        if args.check:
            if path.read_text(encoding="utf-8") != content:
                raise ValueError(f"generated file mismatch: {name}")
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    print("RTS-24 conversion verified" if args.check else "RTS-24 inputs generated")


if __name__ == "__main__":
    main()
