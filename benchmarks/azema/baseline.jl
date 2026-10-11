# Original 3-bin master/oracle/cut policies with common initialization and deadline-safe evidence.
function run_baseline()
    initial = job["initial"]
    budget = Float64(job["time_limit"])
    charged = Float64(initial["charged_elapsed_seconds"])
    elapsed() = (time_ns()-process_started_ns)/1e9+charged+
        (process_started_unix-get(job,"launch_epoch_seconds",process_started_unix))
    remaining() = budget-elapsed()
    directory = job["directory"]
    mkpath(directory)
    best_upper = Float64(initial["objective"])
    best_lower = Float64(initial["lower_bound"])
    best_u = from_rows(initial["u"])
    bound_source = Dict{String,Any}("kind"=>"common_relaxation", "value"=>best_lower)
    job["author_upper"] = best_upper
    last_log = -Inf
    iterations, cuts_count = 0, 0

    function append_record(name, row)
        open(joinpath(directory, name), "a") do io
            JSON3.write(io, row); write(io, '\n'); flush(io)
        end
    end
    function record(event; force=false)
        seconds = elapsed()
        if seconds > budget || (!force && seconds-last_log < 1.)
            return
        end
        best_lower <= best_upper+1e-5 || error("Author bound exceeds the validated upper bound")
        last_log = seconds
        row = Dict("event"=>event, "elapsed_seconds"=>seconds, "objective"=>best_upper,
            "lower_bound"=>best_lower, "relative_gap"=>(best_upper-best_lower)/max(1.,abs(best_upper)),
            "sa_calls"=>0, "llm_calls"=>0)
        checkpoint = "checkpoints/author-$(time_ns()).json"
        value = merge(row, Dict("u"=>matrix_rows(best_u), "bound_source"=>copy(bound_source),
            "model_fingerprint"=>initial["model_fingerprint"], "iterations"=>iterations, "cuts"=>cuts_count))
        path = joinpath(directory,checkpoint)
        save_json(path,value)
        persisted = elapsed()
        if persisted <= budget
            row["elapsed_seconds"] = persisted
            row["checkpoint"] = checkpoint
            row["checkpoint_sha256"] = bytes2hex(sha256(read(path)))
            append_record("trace.jsonl",row)
            save_json(joinpath(directory,"checkpoint.json"),merge(value,row))
        end
    end
    record("initialization"; force=true)
    instance = BDSUC.parse_nc4("Data/T-Ramp/100_0_$(job["base"])_w.nc4", optimizer, 24)
    options = BDSUC.binBD
    N, T, S = instance.N, instance.TimeHorizon, Int(job["scenarios"])
    m = options.master_problem(instance; silent=true, S=S)
    oracle = options.oracle_problem(instance)
    for (key, val) in ["Seed"=>job["seed"], "Threads"=>1, "FeasibilityTol"=>1e-8,
                       "IntFeasTol"=>1e-9, "OptimalityTol"=>1e-8]
        set_optimizer_attribute(m, key, val)
    end
    set_optimizer_attribute(m, "LogFile", joinpath(directory,"gurobi.log"))
    set_optimizer_attribute(m, "LazyConstraints", 1)
    set_optimizer_attribute(m, "MIPGap", job["tolerance"])
    @constraint(m, m[:thermal_fixed_cost]+m[:thermal_cost] >= best_lower)
    u0 = [Float64(unit.InitUpDownTime>=0) for unit in instance.Thermalunits]
    for i in 1:N
        set_start_value(m[:is_on][i,0], u0[i])
        for t in 1:T
            previous = t == 1 ? u0[i] : best_u[i,t-1]
            set_start_value(m[:is_on][i,t], best_u[i,t])
            set_start_value(m[:start_up][i,t], max(0., best_u[i,t]-previous))
            set_start_value(m[:start_down][i,t], max(0., previous-best_u[i,t]))
        end
    end
    set_start_value(m[:thermal_fixed_cost], initial["fixed_cost"])
    set_start_value(m[:thermal_cost], best_upper-initial["fixed_cost"])
    for s in 1:S
        set_start_value(m[:thermal_fuel_cost][s], initial["scenario_recourse"][s])
    end

    sparse_values(a) = Dict("indices"=>findall(!iszero,vec(a)), "values"=>filter(!iszero,vec(a)))
    function save_cut(parameters, optimality)
        selected = optimality ? parameters : [first(p for p in parameters if !p.status)]
        beta = zeros(3*N*T)
        alpha = 0.
        for p in selected
            alpha += p.intercept3bin
            for unit in instance.Thermalunits, t in 1:T
                i = unit.name; idx = (t-1)*N+i
                pm, ru, rd = unit.MinPower, unit.DeltaRampUp, unit.DeltaRampDown
                beta[idx] += p.mumin[i,t]*pm-p.mumax[i,t]*unit.MaxPower-
                             p.muup[i,t]*(pm+ru)+p.mudown[i,t]*pm
                beta[N*T+idx] += p.muup[i,t]*ru
                beta[2*N*T+idx] += p.mudown[i,t]*rd
                previous = p.muup[i,t]*pm-p.mudown[i,t]*(pm+rd)
                if t == 1
                    alpha += previous*u0[i]
                else
                    beta[idx-N] += previous
                end
            end
        end
        beta ./= length(selected); alpha /= length(selected)
        cuts_count += 1
        proof = [Dict("scenario"=>findfirst(q -> q === p,parameters),
            "mumin"=>sparse_values(p.mumin), "mumax"=>sparse_values(p.mumax),
            "muup"=>sparse_values(p.muup), "mudown"=>sparse_values(p.mudown),
            "nu"=>sparse_values(p.ν)) for p in selected]
        append_record("certificate_cuts.jsonl", Dict("cut_id"=>"author-$cuts_count", "source"=>"author_3bin",
            "optimality"=>optimality, "alpha"=>alpha, "beta"=>beta, "proof"=>proof, "elapsed_seconds"=>elapsed()))
    end
    function feasible_callback(cb, master, instance, parameters, gamma, intervals; S)
        added = options._add_feasibility_cuts(cb,master,instance,parameters,gamma,intervals;S=S)
        if added
            save_cut(parameters,false)
        end
        return added
    end
    function optimal_callback(cb,master,instance,parameters,gamma,intervals;S)
        options._add_optimality_cuts(cb,master,instance,parameters,gamma,intervals;S=S)
        save_cut(parameters,true)
    end
    function timed_second_stage(instance, oracle; kwargs...)
        remaining() > 0 || throw(InterruptException())
        set_optimizer_attribute(oracle,"TimeLimit",min(15.,remaining()))
        result = options.second_stage(instance,oracle;kwargs...)
        remaining() > 0 || throw(InterruptException())
        status = termination_status(oracle)
        status in (MOI.OPTIMAL,MOI.DUAL_INFEASIBLE) || error("Uncertified author oracle status: $status")
        return result
    end
    tracked_options = BDSUC.Options(options.master_problem,options.oracle_problem,timed_second_stage,
                                   options.add_cut,feasible_callback,optimal_callback)
    times = []
    function callback(cb, where::Cint)
        if remaining() <= 0
            Gurobi.GRBterminate(cb.model)
            return
        end
        if where == Gurobi.GRB_CB_MIP || where == Gurobi.GRB_CB_MIPSOL
            code = where == Gurobi.GRB_CB_MIP ? Gurobi.GRB_CB_MIP_OBJBND : Gurobi.GRB_CB_MIPSOL_OBJBND
            bound = Ref{Cdouble}()
            Gurobi.GRBcbget(cb, where, code, bound) == 0 || error("Cannot read author master bound")
            if isfinite(bound[]) && bound[] < 1e90
                if bound[]-1e-5 > best_lower
                    best_lower = bound[]-1e-5
                    bound_source = Dict("kind"=>"author_master", "value"=>best_lower,
                        "raw_bound"=>bound[], "cuts"=>cuts_count, "callback"=>Int(where))
                end
            end
            record("master")
            if best_upper-best_lower <= 1e-5+job["tolerance"]*max(1.,abs(best_upper))
                record("converged";force=true)
                Gurobi.GRBterminate(cb.model)
                return
            end
        end
        if where != Gurobi.GRB_CB_MIPSOL
            return
        end
        iterations += 1
        Gurobi.load_callback_variable_primal(cb,where)
        on = convert(Matrix{Float64},round.(callback_value.(cb,m[:is_on])))
        up = convert(Matrix{Float64},round.(callback_value.(cb,m[:start_up])))
        down = convert(Matrix{Float64},round.(callback_value.(cb,m[:start_down])))
        actual, scenario_costs = options.add_cut(cb,tracked_options,m,oracle,instance,times,[on,up,down];
            force=1., S=S, batch=job["batch"], gap=100*job["tolerance"])
        if remaining() <= 0
            return
        end
        if isfinite(actual)
            fixed = sum(unit.ConstTerm*on[unit.name,t+1]+unit.StartUpCost*up[unit.name,t]+
                unit.StartDownCost*down[unit.name,t] for unit in instance.Thermalunits for t in 1:T)
            exact = fixed+sum(scenario_costs)/S
            append_record("evaluations.jsonl",Dict("source"=>"author_3bin", "u"=>matrix_rows(on[:,2:end]),
                "objective"=>exact,"fixed_cost"=>fixed,"feasible"=>true,"elapsed_seconds"=>elapsed()))
            if exact < best_upper
                best_upper = exact; best_u = on[:,2:end]
            end
            # Original 0.999999 improvement cutoff and heuristic submission after ten callbacks.
            if actual <= 0.999999*get(job,"author_upper",1e9)
                job["author_upper"] = actual
                cstr = @build_constraint(m[:thermal_cost]+m[:thermal_fixed_cost] <= actual)
                MOI.submit(m,MOI.LazyConstraint(cb),cstr)
                if iterations >= 10
                    vars = vcat([m[:is_on][i,t] for i in 1:N for t in 0:T],
                        [m[:start_up][i,t] for i in 1:N for t in 1:T],
                        [m[:start_down][i,t] for i in 1:N for t in 1:T],
                        [m[:thermal_fixed_cost],m[:thermal_cost]],[m[:thermal_fuel_cost][s] for s in 1:S])
                    vals = vcat([on[i,t+1] for i in 1:N for t in 0:T],
                        [up[i,t] for i in 1:N for t in 1:T],[down[i,t] for i in 1:N for t in 1:T],
                        [callback_value(cb,m[:thermal_fixed_cost]),sum(scenario_costs)/S],scenario_costs)
                    MOI.submit(m,MOI.HeuristicSolution(cb),vars,vals)
                end
            end
            record("evaluation";force=true)
        end
    end
    MOI.set(m,Gurobi.CallbackFunction(),callback)
    set_optimizer_attribute(m,"TimeLimit",max(.001,remaining()))
    if best_upper-best_lower > 1e-5+job["tolerance"]*max(1.,abs(best_upper)) && remaining()>0
        optimize!(m)
        if remaining()>0
            if isfinite(objective_bound(m)) && objective_bound(m)-1e-5 > best_lower
                best_lower = objective_bound(m)-1e-5
                bound_source = Dict("kind"=>"author_master", "value"=>best_lower,
                    "raw_bound"=>objective_bound(m), "cuts"=>cuts_count,
                    "status"=>string(termination_status(m)))
            end
            record("final";force=true)
        end
    end
    status = best_upper-best_lower <= 1e-5+job["tolerance"]*max(1.,abs(best_upper)) ? "CONVERGED" :
        (remaining() <= 0 ? "TIME_LIMIT" : "SOLVER_STOPPED")
    checkpoint = JSON3.read(read(joinpath(directory,"checkpoint.json"),String),Dict{String,Any})
    save_json(joinpath(directory,"result.json"),merge(checkpoint,Dict("status"=>status,
        "elapsed_seconds"=>elapsed(),"iterations"=>iterations,"cuts"=>cuts_count,
        "common_initialization_sha256"=>initial["sha256"],"scenario_subproblem_seconds"=>times,
        "gurobi_version"=>string(MOI.get(backend(m),MOI.SolverVersion())))))
end
