import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import gurobipy as gp

from suc_benders.candidates import make_candidate
from suc_benders.data import Scenario
from suc_benders.rts24_experiment import read_events
from suc_benders.llm import LLMConfig, LLMAdviceResult, ask_llm_for_candidates
from suc_benders.paper_model import CompactQubo
from suc_benders.solver import BendersConfig, solve_benders
from tests.test_capacity import one_hour_data
from tests.test_paper import candidate_point


def configuration(seed, strategy, rounds=50):
    return BendersConfig(max_iterations=rounds, stop_on_no_new_candidates=False,
        llm=LLMConfig(enabled=strategy == "sa_llm", api_format="chat_completions",
            base_url="https://api.deepseek.com", model="deepseek-flash", timeout_seconds=None,
            max_retries=1, collect_diagnostics=True))


class DynamicSolverTests(unittest.TestCase):
    def test_fixed_rounds_do_not_stop_on_duplicates_and_remove_llm_timeout(self):
        data=replace(one_hour_data(),base_load=np.full(3,30.))
        ss=[Scenario("tiny",1.,data.base_load)]
        events=[]
        config=replace(configuration(7,"sa_llm",rounds=3),event_sink=events.append)
        def advise(*args, **kwargs):
            self.assertIsNone(args[5].timeout_seconds)
            return LLMAdviceResult([])
        with patch("suc_benders.paper_solver.solve_qubo",return_value=[]) as sa, \
             patch("suc_benders.paper_solver.ask_llm_for_candidates",side_effect=advise) as llm:
            solve_benders(data,ss,config)
        self.assertEqual(sa.call_count,3)
        self.assertEqual(llm.call_count,3)
        self.assertEqual(sum(e['kind']=='iteration' for e in events),3)
        self.assertIsNone(config.time_limit_seconds)
        self.assertFalse(config.stop_on_no_new_candidates)

    def test_unlimited_http_timeout_and_no_retry_on_server_error(self):
        data=one_hour_data()
        with patch.dict("os.environ",{"LLM_API_KEY":"SECRET"}), \
             patch("suc_benders.llm.build_opener") as opener:
            opener.return_value.open.side_effect=ConnectionError()
            result=ask_llm_for_candidates(data,[Scenario("tiny",1.,data.base_load)],1,[],None,
                                         configuration(7,"sa_llm",rounds=50).llm)
        self.assertTrue(result.failed)
        self.assertIsNone(opener.return_value.open.call_args.kwargs['timeout'])
        self.assertEqual(opener.return_value.open.call_count,1)


    def test_partial_last_event_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"events.jsonl"
            p.write_text('{"kind":"sa_start"}\n{"ki', encoding="utf-8")
            self.assertEqual(read_events(p), [{"kind":"sa_start"}])

    def test_requested_api_settings_and_scrubbing(self):
        d = one_hour_data()
        ss = [Scenario("tiny", 1., d.base_load)]
        captured = {}
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps({"model":"test", "choices":[{"finish_reason":"stop",
                    "message":{"content":'{"candidates":[]}', "reasoning_content":"SECRET"}}]}).encode()
        class Opener:
            def open(self, request, timeout):
                captured.update(json.loads(request.data))
                return Response()
        with patch.dict("os.environ", {"LLM_API_KEY":"SECRET"}), patch("suc_benders.llm.build_opener", return_value=Opener()):
            r = ask_llm_for_candidates(d, ss, 1, [], None, configuration(7,"sa_llm").llm)
        self.assertFalse(r.failed)
        self.assertEqual(captured["reasoning_effort"], "max")
        self.assertEqual(captured["max_tokens"], 65536)
        self.assertNotIn("SECRET", json.dumps(r.diagnostics))

    def test_no_retry_on_timeout(self):
        d = one_hour_data()
        with patch.dict("os.environ", {"LLM_API_KEY":"SECRET"}), patch("suc_benders.llm.build_opener") as opener:
            opener.return_value.open.side_effect = TimeoutError()
            result = ask_llm_for_candidates(d, [Scenario("tiny",1.,d.base_load)], 1, [], None, configuration(7,"sa_llm").llm)
        self.assertTrue(result.failed)
        self.assertEqual(opener.return_value.open.call_count, 1)

    def test_dynamic_llm_candidates_share_cuts_and_validation(self):
        data = one_hour_data()
        vectors = {k:np.repeat(v,2) for k,v in vars(data).items() if isinstance(v,np.ndarray) and k != "base_load"}
        data = replace(data, **vectors, n_units=2, base_load=np.full(3, 30.))
        scenarios = [Scenario("tiny",1.,data.base_load)]
        seed = make_candidate(data, scenarios, np.ones((2,3),dtype=int), source="fallback")
        proposal = make_candidate(data, scenarios, np.array([[1,1,1],[0,0,0]]), source="llm")
        events, contexts = [], []
        def advise(*args, **kwargs):
            contexts.append(kwargs["context"])
            self.assertGreater(args[5].timeout_seconds, 180)
            self.assertLessEqual(args[5].timeout_seconds, 300)
            return LLMAdviceResult([proposal, proposal])
        config = replace(configuration(7,"sa_llm"), max_iterations=2,
                         time_limit_seconds=400, llm=replace(configuration(7,"sa_llm").llm, timeout_seconds=300), event_sink=events.append)
        optimize = gp.Model.optimize
        def continuous(model, *args, **kwargs):
            model.update()
            self.assertTrue(all(v.VType == gp.GRB.CONTINUOUS for v in model.getVars()))
            return optimize(model,*args,**kwargs)
        with patch("suc_benders.solver._construct_feasible_candidate",return_value=seed), \
             patch("suc_benders.paper_solver.solve_qubo",return_value=[np.array([])]), \
             patch.object(CompactQubo,"decode",return_value=(candidate_point(seed),0.)), \
             patch("suc_benders.paper_solver.ask_llm_for_candidates",side_effect=advise), \
             patch.object(gp.Model,"optimize",continuous):
            result = solve_benders(data,scenarios,config)
        self.assertTrue(contexts)
        self.assertNotIn("decommit_ladder",contexts[0])
        self.assertEqual(sum(e["kind"]=="evaluation" and e["source"]=="llm" for e in events),1)
        self.assertEqual(result.diagnostics["local_evaluation_count"],2)
        self.assertEqual(result.optimality_cuts,2)
        self.assertEqual(int(result.u.sum()),3)
        self.assertEqual(result.diagnostics["sa_calls"],sum(e["kind"]=="sa_start" for e in events))

    def test_no_new_candidates_keeps_running_until_deadline(self):
        data=replace(one_hour_data(),base_load=np.full(3,30.))
        ss=[Scenario("tiny",1.,data.base_load)]
        events=[]
        config=replace(configuration(7,"sa",rounds=100000),time_limit_seconds=.3,event_sink=events.append)
        with patch("suc_benders.paper_solver.solve_qubo",return_value=[]):
            result=solve_benders(data,ss,config)
        self.assertEqual(result.status,"TIME_LIMIT")
        self.assertGreater(sum(e["kind"]=="sa_start" for e in events),1)


if __name__ == "__main__":
    unittest.main()
