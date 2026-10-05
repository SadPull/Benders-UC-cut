import itertools
import tempfile
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import gurobipy as gp
import numpy as np

from suc_benders.block_master_model import (RecordedQubo, continuous_relaxation, fixed_master_value,
    partition_rows, priced_blocks, solve_master)
from suc_benders.block_master_experiment import (aggregate, audit_encoding, benchmark_snapshot,
    classical_trace, evaluate_sample, relative_gap, repair_sample)
from suc_benders.candidates import make_candidate
from suc_benders.data import Scenario
from suc_benders.full_model import FullConfig, solve_full
from suc_benders.paper_model import IndexedUC, CompactQubo, Cut, binary, key
from suc_benders.solver import BendersConfig
from tests.test_capacity import one_hour_data


def point(candidate, keys):
    return {k: int(getattr(candidate, k[0])[k[1], k[2]]) for k in keys}


def merged_optimum(indexed, blocks, cross):
    model = gp.Model("merged_partition_test")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.MIPGap = 1e-9
    variables = {k: model.addVar(lb=lo, ub=hi, vtype=gp.GRB.BINARY if binary(k) else gp.GRB.CONTINUOUS)
                 for k, (lo, hi) in indexed.bounds.items()}
    for row in [r for b in blocks for r in b.rows] + [indexed.rows[i] for i in cross]:
        expression = gp.quicksum(v*variables[k] for k, v in row.a.items()) + row.c
        model.addConstr(expression == 0 if row.equality else expression <= 0)
    model.setObjective(gp.quicksum(indexed.linear[k]*v+indexed.quadratic.get(k,0.)*v*v for k,v in variables.items()))
    model.optimize()
    value = model.ObjVal
    model.dispose()
    return value


class BlockMasterTests(unittest.TestCase):
    def setUp(self):
        self.data = replace(one_hour_data(), base_load=np.full(3,30.))
        self.scenarios = [Scenario("low", .3, np.full(3,25.)), Scenario("base",.4,np.full(3,30.)),
                          Scenario("high",.3,np.full(3,35.))]
        self.indexed = IndexedUC(self.data, self.scenarios)

    def test_recombined_partition_matches_original_full_model(self):
        data = replace(self.data, min_up=np.array([3]), min_down=np.array([4]), initial_duration=np.array([4.]))
        indexed = IndexedUC(data,self.scenarios)
        blocks,cross = priced_blocks(indexed,1,[0.]*len(indexed.rows))
        ids = [i for b in blocks for i in b.row_ids]+cross
        self.assertEqual(sorted(ids),list(range(len(indexed.rows))))
        original = solve_full(data,self.scenarios,FullConfig(time_limit=30,mip_gap=1e-9,output_flag=0))
        self.assertAlmostEqual(merged_optimum(indexed,blocks,cross),original.objective,places=5)
        original.model.dispose()
        self.assertEqual(sum(len(b.bkeys) for b in blocks),3*data.n_units*data.horizon)
        labels={indexed.rows[i].label for i in cross}
        self.assertTrue({'transition','min_up','min_down','ramp_up','ramp_down','shutdown_capacity'} <= labels)

    def test_internal_boundaries_do_not_reset_initial_or_terminal_rules(self):
        data=replace(self.data,horizon=6,base_load=np.full(6,30.),min_up=np.array([4]),min_down=np.array([3]),
                     initial_on=np.array([True]),initial_duration=np.array([1.]),initial_output=np.array([30.]))
        indexed=IndexedUC(data,[Scenario('single',1.,data.base_load)])
        blocks,cross=priced_blocks(indexed,2,[0.]*len(indexed.rows))
        self.assertEqual([b.times for b in blocks],[(0,1),(2,3),(4,5)])
        terminal=[k[2] for b in blocks for r in b.rows if r.label=='terminal' for k in r.a]
        initial=[k[2] for b in blocks for r in b.rows if r.label=='initial' for k in r.a]
        self.assertEqual(terminal,[5])
        self.assertEqual(initial,[0,1,2])
        self.assertTrue(any(indexed.rows[i].label=='min_up' and max(k[2] for k in indexed.rows[i].a)>=3 for i in cross))

    def test_lagrangian_objective_identity_and_multiplier_signs(self):
        local,cross=partition_rows(self.indexed,1)
        prices=[0.]*len(self.indexed.rows)
        for i in cross:
            prices[i]=-2.5 if self.indexed.rows[i].equality else .7
        blocks,_=priced_blocks(self.indexed,1,prices)
        rng=np.random.default_rng(42)
        values={k:float(rng.uniform(lo,hi)) for k,(lo,hi) in self.indexed.bounds.items()}
        original=sum(self.indexed.linear[k]*v+self.indexed.quadratic.get(k,0.)*v*v for k,v in values.items())
        lagrangian=original+sum(prices[i]*self.indexed.rows[i].value(values) for i in cross)
        summed=sum(b.constant+sum(b.costs[k]*values[k]+self.indexed.quadratic.get(k,0.)*values[k]**2 for k in b.keys) for b in blocks)
        self.assertAlmostEqual(summed,lagrangian,places=8)
        bad=next(i for i in cross if not self.indexed.rows[i].equality)
        prices[bad]=-1.
        with self.assertRaises(ValueError): priced_blocks(self.indexed,1,prices)

    def test_qp_prices_have_correct_sign_and_preserve_weak_duality(self):
        result=continuous_relaxation(self.indexed,time.perf_counter()+30)
        self.assertTrue(all(p>=0 for p,r in zip(result['multipliers'],self.indexed.rows) if not r.equality))
        blocks,cross=priced_blocks(self.indexed,1,result['multipliers'])
        original=solve_full(self.data,self.scenarios,FullConfig(time_limit=30,mip_gap=1e-9,output_flag=0))
        feasible_point={key(k,0,t):int(getattr(original,k)[0,t]) for k in ('u','v','w') for t in range(3)}
        for s,sol in enumerate(original.scenarios):
            for kind in ('p','r'):
                for t in range(3): feasible_point[key(kind,0,t,s)]=float(getattr(sol,kind)[0,t])
        penalty=sum(result['multipliers'][i]*self.indexed.rows[i].value(feasible_point) for i in cross)
        self.assertLessEqual(penalty,1e-5)
        self.assertLessEqual(result['objective'],original.objective+1e-5)
        original.model.dispose()

    def test_price_specific_dispatch_cuts_support_all_local_feasible_points(self):
        _,cross=partition_rows(self.indexed,1)
        prices=[0.]*len(self.indexed.rows)
        ramp=next(i for i in cross if self.indexed.rows[i].label=='ramp_up' and key('p',0,0,0) in self.indexed.rows[i].a)
        prices[ramp]=10.
        zero,_=priced_blocks(self.indexed,1,[0.]*len(prices))
        changed,_=priced_blocks(self.indexed,1,prices)
        block=changed[0]
        records=[]
        for bits in itertools.product((0,1),repeat=len(block.bkeys)):
            x=dict(zip(block.bkeys,bits))
            if not block.discrete_checks(x)['feasible']: continue
            value=block.dispatch(x,block.costs,time.perf_counter()+30)
            if value and value['feasible']: records.append((x,value))
        self.assertTrue(records)
        for x,value in records:
            self.assertAlmostEqual(value['cut'].value(x),value['recourse'],places=5)
            for xx,other in records:
                self.assertLessEqual(value['cut'].value(xx),other['recourse']+1e-5)
        x,new=records[0]
        old=zero[0].dispatch(x,zero[0].costs,time.perf_counter()+30)
        self.assertGreater(old['cut'].value(x),new['recourse']+1)
        oracle=solve_master(block,[v['cut'] for _,v in records],time.perf_counter()+30)
        self.assertTrue(oracle['certified'])
        self.assertAlmostEqual(fixed_master_value(block,[v['cut'] for _,v in records],oracle['point']),oracle['objective'],places=5)

    def test_recording_adapter_preserves_existing_qubo(self):
        blocks,_=priced_blocks(self.indexed,3,[0.]*len(self.indexed.rows))
        b=blocks[0]
        candidate=make_candidate(self.data,self.scenarios,np.ones((1,3),dtype=int))
        local=b.dispatch(point(candidate,b.bkeys),b.costs,time.perf_counter()+30)
        args=(b,b.costs,[local['cut']],0.,local['cost'],BendersConfig())
        original=CompactQubo(*args)
        recorded=RecordedQubo(*args)
        order=[recorded.names.index(n) for n in original.names]
        np.testing.assert_allclose(original.matrix,recorded.matrix[np.ix_(order,order)],rtol=1e-12,atol=1e-12)
        self.assertAlmostEqual(original.model.get_offset(),recorded.model.get_offset(),places=6)

    def tiny_encoding(self):
        d=replace(one_hour_data(),horizon=1,base_load=np.array([0.]),pmin=np.array([0.]),pmax=np.array([1.]),
            pstart=np.array([1.]),pshut=np.array([1.]),cost_a=np.array([0.]),cost_b=np.array([0.]),
            cost_c=np.array([1.]),startup_cost=np.array([0.]),shutdown_cost=np.array([0.]))
        scenarios=[Scenario('tiny',1.,d.base_load)]
        indexed=IndexedUC(d,scenarios)
        block=priced_blocks(indexed,1,[0.]*len(indexed.rows))[0][0]
        cuts=[Cut({},.15)]
        master=RecordedQubo(block,block.costs,cuts,0.,2.,BendersConfig())
        return d,scenarios,block,cuts,master

    def test_exhaustive_tiny_qubo_matches_declared_quantization_and_energy(self):
        d,scenarios,b,cuts,m=self.tiny_encoding()
        self.assertLessEqual(len(m.names),15)
        oracle=solve_master(b,cuts,time.perf_counter()+30)
        energies=[]
        for assignment in itertools.product((0,1),repeat=len(m.names)):
            bits=np.asarray(assignment)
            energy=float(bits@m.matrix@bits+m.model.get_offset())
            values=dict(zip(m.names,bits))
            polynomial=0.
            for constraint in m.model.hard_constraints_made.values():
                expr=constraint.constraint_expr
                residual=expr.offset+sum(coef*np.prod([values[v] for v in term]) for term,coef in expr.coefficient.items())
                polynomial += 1e6*residual
            _,theta=m.decode(bits)
            x,_=m.decode(bits)
            objective=theta+sum(b.costs[k]*x[k] for k in b.bkeys)
            self.assertAlmostEqual(energy,objective/b.scale+polynomial,places=5)
            energies.append((energy,bits))
        _,bits=min(energies,key=lambda p:p[0])
        self.assertTrue(m.encoded_checks(bits)['feasible'])
        decoded,theta=m.decode(bits)
        value=theta+sum(b.costs[k]*decoded[k] for k in b.bkeys)+b.constant
        self.assertLessEqual(abs(value-oracle['objective']),1.)
        initial=make_candidate(d,scenarios,np.zeros((1,1),dtype=int))
        sample=evaluate_sample(m,b,cuts,bits,initial,oracle,7,0)
        self.assertTrue(sample['raw_encoded_feasible'])
        self.assertFalse(sample['raw_native_master_feasible'])
        self.assertIsNone(sample['raw_gap'])
        self.assertAlmostEqual(sample['classical_fixed_raw_gap'],0.)
        self.assertTrue(all(row['rounding_error']<=1. for row in m.inequalities))

    def test_repair_uses_original_full_horizon_and_records_outside_changes(self):
        b=priced_blocks(self.indexed,1,[0.]*len(self.indexed.rows))[0][-1]
        initial=make_candidate(self.data,self.scenarios,np.ones((1,3),dtype=int))
        x=point(initial,b.bkeys)
        x[key('u',0,2)]=0
        repaired,changes,outside=repair_sample(b,x,initial)
        self.assertEqual(changes,1)
        self.assertEqual(outside,0)
        self.assertEqual(repaired[key('u',0,2)],1)
        self.assertEqual(initial.u.tolist(),[[1,1,1]])

    def test_snapshot_witness_is_audited_but_not_used_as_sa_start(self):
        from pathlib import Path
        d,scenarios,b,cuts,master=self.tiny_encoding()
        initial=make_candidate(d,scenarios,np.zeros((1,1),dtype=int))
        reference=solve_master(b,cuts,time.perf_counter()+30)
        snapshot=dict(cuts=cuts,lower=0.,upper=2.,upper_point=point(initial,b.bkeys),reference=reference)
        self.assertTrue(audit_encoding(master,b,snapshot)['passed'])
        with tempfile.TemporaryDirectory() as directory, patch('suc_benders.block_master_experiment.solve_qubo',
            return_value=[np.zeros(len(master.names),dtype=int)]) as sampler:
            result=benchmark_snapshot(Path(directory),b,snapshot,initial,BendersConfig(),time.perf_counter()+30)
        self.assertTrue(result['complete'])
        self.assertEqual(sampler.call_count,3)
        for call in sampler.call_args_list:
            self.assertEqual(set(call.kwargs),{'energy_unit'})
        row=dict(reference=reference,samples=result,encoding={})
        summary=aggregate([row])
        self.assertEqual(summary['full_master_feasible'],0)
        self.assertEqual(summary['certified_with_5pct_unmodified_commitment'],1)
        self.assertTrue(all(s['raw_gap'] is None for s in result['samples']))

    def test_bad_encoding_stops_before_sa(self):
        d,scenarios,b,cuts,master=self.tiny_encoding()
        initial=make_candidate(d,scenarios,np.zeros((1,1),dtype=int))
        snapshot=dict(cuts=[Cut({},100.)],upper_point=point(initial,b.bkeys))
        with self.assertRaises(ValueError): audit_encoding(master,b,snapshot)

    def test_reference_gap_requires_certification_and_handles_negative_cost(self):
        self.assertIsNone(relative_gap(12.,dict(certified=False,objective=10.)))
        self.assertAlmostEqual(relative_gap(-8.,dict(certified=True,objective=-10.)),.2)

    def test_snapshot_bounds_do_not_use_current_integer_optimum(self):
        b=SimpleNamespace(costs={},bkeys=[],constant=0.,discrete_checks=lambda x:dict(feasible=True))
        dispatches=[dict(feasible=True,cost=c,cut=Cut({},c)) for c in (100.,80.,40.)]
        b.dispatch=lambda *args:dispatches.pop(0)
        references=[]
        for value in (10.,20.,30.,40.):
            references.append(dict(certified=True,objective=value,lower_bound=value,point={},status=2,theta=value))
        with patch('suc_benders.block_master_experiment.solve_master',side_effect=references):
            states,_,status=classical_trace(b,None,time.perf_counter()+30,lambda i:None)
        self.assertEqual(status,'converged')
        self.assertEqual(states['initial']['upper'],100.)
        self.assertEqual(states['final']['upper'],80.)
        self.assertAlmostEqual(states['initial']['lower'],10.-1e-5)


if __name__=='__main__':
    unittest.main()
