import json
from pathlib import Path
import runpy
from types import SimpleNamespace as NS

import numpy as np
import pytest
import run_suc
from suc_benders.solver import Result


def test_file_settings_run_from_another_directory_without_overwriting(tmp_path, monkeypatch, capsys):
    seen = []
    references = []

    def solve(options):
        seen.append(options)
        return Result(status="TIME_LIMIT", objective=110., lower_bound=100., relative_gap=10./110.,
                      validation={"feasible": True}, diagnostics={"sa_calls": 1})

    monkeypatch.setattr(run_suc, "run_suc", solve)
    entry = Path(__file__).resolve().parents[1] / "run_experiment.py"
    module = runpy.run_path(str(entry), run_name="import_check")
    assert not seen
    main = module["main"]

    def reference(data, scenarios, seconds, log_file):
        references.append((data.n_units, len(scenarios), seconds, log_file))
        return dict(status="TIME_LIMIT", objective=105., lower=104., gap_percent=100./105.,
                    validation={"feasible": True}, elapsed_seconds=1., u=[[1]], v=[[0]],
                    w=[[0]], scenarios=[])

    project = tmp_path / "project"
    folder = project / "data"
    folder.mkdir(parents=True)
    main.__globals__.update(PROJECT=project, SEED=29, TIME_LIMIT_SECONDS=60,
                            TARGET_GAP_PERCENT=.1, RESULTS=tmp_path / "results",
                            solve_reference=reference)
    monkeypatch.chdir(tmp_path)
    for source in (entry.parent / "data", entry.parent / "tests/fixtures/rts24"):
        for name in ("units.json", "stochastic.json"):
            (folder / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
        assert main() == 0
    assert len(seen) == 2
    assert [(n, s, seconds) for n, s, seconds, _ in references] == [(10, 3, 120), (32, 3, 120)]
    assert all(o["seed"] == 29 and o["time_limit"] == 60 and o["tolerance"] == .001
               and o["llm_enabled"] is True and o["llm_model"] == "gpt-6-astra"
               and o["llm_codex_command"] == "codex" and o["llm_timeout"] == 1200
               and o["llm_reserve_seconds"] == 60 for o in seen)
    outputs = list((tmp_path / "results").iterdir())
    assert len(outputs) == 2
    dimensions = set()
    for output in outputs:
        assert output.name.startswith("manual-seed29-")
        options = json.loads((output / "options.json").read_text(encoding="utf-8"))
        assert Path(options["data_file"]) == folder / "units.json"
        assert Path(options["scenario_file"]) == folder / "stochastic.json"
        assert Path(options["llm_agent_directory"]) == output / "llm_requests"
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert "case" not in summary
        dimensions.add((summary["units"], summary["hours"], summary["scenarios"]))
        assert summary["lower"] == 100.
        assert summary["reference"]["objective"] == 105.
        assert summary["reference"]["lower"] == 104.
        assert summary["reference_gap_percent"] == pytest.approx(100 * 5. / 105.)
        reference_result = json.loads((output / "reference.json").read_text(encoding="utf-8"))
        assert reference_result["u"] == [[1]]
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        assert result["coordination_upper"] is None
    assert dimensions == {(10, 24, 3), (32, 24, 3)}
    console = capsys.readouterr().out
    assert f"机组输入：{folder / 'units.json'}" in console
    assert f"场景输入：{folder / 'stochastic.json'}" in console
    assert "输入规模：10机组；24小时；3场景" in console
    assert "输入规模：32机组；24小时；3场景" in console
    assert "主算法可行成本（UB）：110.000000" in console
    assert "认证下界（LB）：100.000000" in console
    assert "Gurobi 完整问题参考成本：105.000000" in console
    assert "主算法 UB 相对 Gurobi 参考成本的差距：4.761905%" in console
    assert "尚未证明最优" in console
    assert "调用方式：Codex 独立任务" in console


def test_check_only_does_not_start_solver_or_create_results(tmp_path, monkeypatch):
    monkeypatch.setattr(run_suc, "run_suc", lambda options: (_ for _ in ()).throw(AssertionError("Solver started")))
    entry = Path(__file__).resolve().parents[1] / "run_experiment.py"
    module = runpy.run_path(str(entry), run_name="import_check")
    main = module["main"]
    main.__globals__["solve_reference"] = lambda *args: pytest.fail("Reference solver started")
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: pytest.fail("Codex started"))
    main.__globals__.update(CHECK_ONLY=True, RESULTS=tmp_path / "results")
    assert main() == 0
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("status", ["TIME_LIMIT", "PAUSED_LLM_FAILURE"])
def test_llm_can_be_disabled_and_failure_returns_nonzero(tmp_path, monkeypatch, capsys, status):
    seen = []

    def solve(options):
        seen.append(options)
        return Result(status=status, objective=110., lower_bound=100., relative_gap=10./110.,
                      validation={"feasible": True})

    monkeypatch.setattr(run_suc, "run_suc", solve)
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "run_experiment.py"))
    assert module["TIME_LIMIT_SECONDS"] == 1800
    assert module["REFERENCE_TIME_LIMIT_SECONDS"] == 120
    main = module["main"]
    main.__globals__.update(LLM_ENABLED=False, RESULTS=tmp_path / "results", solve_reference=lambda *a:
        dict(status="OPTIMAL", objective=105., lower=105., gap_percent=0., validation={"feasible": True}, elapsed_seconds=0.))
    assert main() == (1 if status == "PAUSED_LLM_FAILURE" else 0)
    assert not seen[0]["llm_enabled"] and not seen[0]["llm_codex_command"]
    assert not seen[0]["llm_agent_directory"]
    output, = (tmp_path / "results").iterdir()
    assert json.loads((output / "result.json").read_text(encoding="utf-8"))["status"] == status
    assert "LLM：关闭" in capsys.readouterr().out


def test_full_gurobi_reference_returns_validated_schedule(tiny, tmp_path):
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "run_experiment.py"))
    d, scenarios = tiny(1)
    reference = module["solve_reference"](d, scenarios, 10, tmp_path / "gurobi.log")
    assert reference["status"] == "OPTIMAL"
    assert reference["objective"] == pytest.approx(225., abs=1e-5)
    assert reference["lower"] == pytest.approx(reference["objective"], abs=1e-4)
    assert reference["gap_percent"] < .001
    assert reference["validation"]["feasible"]
    np.testing.assert_array_equal(reference["u"], [[1, 1, 1]])
    assert len(reference["scenarios"]) == 1


def test_reference_without_incumbent_has_no_cost(tiny, tmp_path, monkeypatch):
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "run_experiment.py"))
    reference = module["solve_reference"]
    monkeypatch.setitem(reference.__globals__, "optimize", lambda *a, **k:
                        NS(status="TIME_LIMIT", lower=-np.inf, x=None))
    d, scenarios = tiny(1)
    result = reference(d, scenarios, 10, tmp_path / "gurobi.log")
    assert result["objective"] is None and result["validation"] is None


@pytest.mark.parametrize("failed", [False, True])
def test_unavailable_reference_preserves_main_result(tmp_path, monkeypatch, capsys, failed):
    monkeypatch.setattr(run_suc, "run_suc", lambda options:
                        Result(status="TIME_LIMIT", objective=110., lower_bound=100.,
                               relative_gap=10./110., validation={"feasible": True}))
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "run_experiment.py"))
    main = module["main"]

    def reference(*args):
        if failed:
            raise ValueError("Physical or total cost validation failed")
        return dict(status="TIME_LIMIT", objective=None, lower=-np.inf, gap_percent=None,
                    validation=None, elapsed_seconds=1.)

    main.__globals__.update(RESULTS=tmp_path / "results", solve_reference=reference)
    assert main() == 1
    output, = (tmp_path / "results").iterdir()
    assert json.loads((output / "result.json").read_text(encoding="utf-8"))["objective"] == 110.
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["reference"]["objective"] is None
    assert summary["reference_gap_percent"] is None
    console = capsys.readouterr().out
    assert "未获得通过校验的参考方案" in console
    assert "主算法 UB 相对 Gurobi 参考成本的差距：无法计算" in console


@pytest.mark.parametrize("upper,reference_cost,feasible,expected_gap", [
    (100., 100., True, 0.),
    (95., 100., True, -5.),
    (.25, .2, True, 25.),
    (110., 0., True, None),
    (110., 100., False, None),
])
def test_reference_gap_handles_equal_lower_small_and_unavailable_costs(
        tmp_path, monkeypatch, capsys, upper, reference_cost, feasible, expected_gap):
    monkeypatch.setattr(run_suc, "run_suc", lambda options:
                        Result(status="TIME_LIMIT", objective=upper,
                               validation={"feasible": True} if feasible else None))
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "run_experiment.py"))
    main = module["main"]
    main.__globals__.update(RESULTS=tmp_path / "results", solve_reference=lambda *a:
        dict(status="TIME_LIMIT", objective=reference_cost, lower=reference_cost, gap_percent=0.,
             validation={"feasible": True}, elapsed_seconds=0.))
    assert main() == int(not feasible)
    output, = (tmp_path / "results").iterdir()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    console = capsys.readouterr().out
    if expected_gap is None:
        assert summary["reference_gap_percent"] is None
        assert "主算法 UB 相对 Gurobi 参考成本的差距：无法计算" in console
    else:
        assert summary["reference_gap_percent"] == pytest.approx(expected_gap)
        assert f"主算法 UB 相对 Gurobi 参考成本的差距：{expected_gap:.6f}%" in console
