# Start before imports so Julia compilation and environment loading count in arm A.
const process_started_ns = time_ns()
const process_started_unix = time()
# Thin experiment driver around the unchanged author's parsing/model/cut functions.
using BDSUC, JuMP, Gurobi, JSON3, LinearAlgebra, SHA
BLAS.set_num_threads(1)

const job = JSON3.read(read(ARGS[1], String), Dict{String, Any})
cd(job["author_root"])

function save_json(path, value)
    mkpath(dirname(path))
    temporary = path * ".tmp"
    open(temporary, "w") do io
        JSON3.write(io, value)
        flush(io)
    end
    mv(temporary, path; force=true)
end

matrix_rows(m) = [collect(m[i, :]) for i in axes(m, 1)]
from_rows(rows) = permutedims(hcat([Float64.(row) for row in rows]...))
optimizer = () -> Gurobi.Optimizer()

function environment_info()
    major, minor, patch = Ref{Cint}(), Ref{Cint}(), Ref{Cint}()
    Gurobi.GRBversion(major,minor,patch)
    return Dict("julia"=>string(VERSION), "gurobi"=>join([major[],minor[],patch[]],"."),
        "Gurobi.jl"=>string(Base.pkgversion(Gurobi)), "JuMP"=>string(Base.pkgversion(JuMP)),
        "julia_threads"=>Threads.nthreads(), "blas_threads"=>BLAS.get_num_threads())
end

function export_inputs()
    instance = BDSUC.parse_nc4("Data/T-Ramp/100_0_$(job["base"])_w.nc4", optimizer, 24)
    units = instance.Thermalunits
    data = Dict{String, Any}("model_profile"=>"azema_sms", "n_units"=>instance.N, "horizon"=>24,
        "shedding_cost"=>BDSUC.SHEDDING_COST, "curtailment_cost"=>BDSUC.CURTAILEMENT_COST,
        "base_load"=>instance.Demandbus[1], "cost_a"=>zeros(instance.N),
        "shutdown_cost"=>zeros(instance.N))
    for (name, field) in ["pmin"=>:MinPower, "pmax"=>:MaxPower, "pstart"=>:MinPower,
        "pshut"=>:MinPower, "ramp_up"=>:DeltaRampUp, "ramp_down"=>:DeltaRampDown,
        "startup_cost"=>:StartUpCost, "cost_b"=>:LinearTerm, "cost_c"=>:ConstTerm,
        "initial_output"=>:InitialPower, "initial_duration"=>:InitUpDownTime,
        "min_up"=>:MinUpTime, "min_down"=>:MinDownTime, "source_cost_a"=>:QuadTerm]
        data[name] = [getfield(unit, field) for unit in units]
    end
    days = instance.Training_set[job["batch"]][1:job["scenarios"]]
    scenarios = [Dict("name"=>"day-$day", "probability"=>1/length(days),
        "load"=>instance.Demandbus[1]-instance.WGscenario[1][:, day]) for day in days]
    save_json(job["output"], Dict("units"=>data, "scenarios"=>Dict("reserve_fraction"=>0,
        "scenarios"=>scenarios), "julia_version"=>string(VERSION), "author_commit"=>
        "5b9cf5ec2795b481f4d65c364be37eacfd0b585b"))
end

function json_instance(data, scenarios)
    N, T, S = data["n_units"], data["horizon"], length(scenarios["scenarios"])
    units = BDSUC.ThermalUnit[]
    for i in 1:N
        pm, pu = data["pmin"][i], data["pmax"][i]
        ru, rd = data["ramp_up"][i], data["ramp_down"][i]
        duration, p0 = Int(data["initial_duration"][i]), data["initial_output"][i]
        mu, md = Int(data["min_up"][i]), Int(data["min_down"][i])
        intervals = BDSUC.set_intervals(mu, duration, p0, pu, pm, rd)
        push!(units, BDSUC.ThermalUnit(i, 1, pm, pu, ru, rd, 0., data["startup_cost"][i],
            data["shutdown_cost"][i], data["cost_b"][i], data["cost_c"][i], p0, duration,
            mu, md, intervals, Int(1+floor((pu-pm)/ru)), Int(1+floor((pu-pm)/rd))))
    end
    load = Float64.(data["base_load"])
    wind = hcat([load-Float64.(s["load"]) for s in scenarios["scenarios"]]...)
    return BDSUC.Instance("independent-check", T, N, units, Dict{Tuple{Int64,Int64},BDSUC.Line}(),
        [Int64[]], [load], [1], [wind], [collect(1:S)], 1:S, optimizer,
        Dict{Tuple{Int64,Int64,Int64},JuMP.Model}())
end

function check_model(data=job["units"], scenarios=job["scenarios"]; write_result=true)
    instance = json_instance(data, scenarios)
    N, T, S = instance.N, instance.TimeHorizon, length(scenarios["scenarios"])
    m = BDSUC.master_3BD_problem(instance; silent=true, S=S)
    @variable(m, power[i in 1:N, t in 0:T, s in 1:S] >= 0)
    @variable(m, power_shedding[b in 1:1, t in 0:T, s in 1:S] >= 0)
    @variable(m, power_curtailement[b in 1:1, t in 0:T, s in 1:S] >= 0)
    BDSUC.thermal_unit_capacity_constraints_scenarios(m, instance, S)
    @constraint(m, [t in 1:T, s in 1:S], sum(power[i,t,s] for i in 1:N)+
        power_shedding[1,t,s]-power_curtailement[1,t,s] == scenarios["scenarios"][s]["load"][t])
    if job["mode"] == "fixed"
        for i in 1:N, t in 1:T
            fix(m[:is_on][i,t], job["u"][i][t]; force=true)
        end
    elseif job["mode"] == "relaxation"
        relax_integrality(m)
    end
    set_optimizer_attribute(m, "TimeLimit", job["time_limit"])
    set_optimizer_attribute(m, "MIPGap", 1e-9)
    set_optimizer_attribute(m, "FeasibilityTol", 1e-8)
    set_optimizer_attribute(m, "OptimalityTol", 1e-8)
    set_optimizer_attribute(m, "Seed", 7)
    set_optimizer_attribute(m, "Threads", 1)
    optimize!(m)
    valid = primal_status(m) == MOI.FEASIBLE_POINT
    result = Dict("status"=>string(termination_status(m)),
        "objective"=>(valid ? objective_value(m) : nothing),
        "lower_bound"=>objective_bound(m), "environment"=>environment_info())
    if valid
        result["u"] = [[value(m[:is_on][i,t]) for t in 1:T] for i in 1:N]
        result["scenarios"] = [Dict("name"=>scenarios["scenarios"][s]["name"],
            "p"=>[[value(power[i,t,s]) for t in 1:T] for i in 1:N],
            "shed"=>[value(power_shedding[1,t,s]) for t in 1:T],
            "curtail"=>[value(power_curtailement[1,t,s]) for t in 1:T]) for s in 1:S]
    end
    if write_result
        save_json(job["output"],result)
    end
    return result
end

function audit_fixed()
    result = Dict{String,Any}("status"=>"OPTIMAL", "objective"=>0., "scenarios"=>Any[],
        "environment"=>environment_info())
    for (index, scenario) in enumerate(job["scenarios"]["scenarios"])
        one = Dict("reserve_fraction"=>0, "scenarios"=>[scenario])
        checked = check_model(job["units"],one;write_result=false)
        checked["status"] == "OPTIMAL" || error("Independent scenario $index did not solve optimally")
        result["objective"] += scenario["probability"]*checked["objective"]
        push!(result["scenarios"],only(checked["scenarios"]))
        save_json(job["output"]*".progress", Dict("scenarios_completed"=>index))
    end
    save_json(job["output"],result)
end

if job["action"] == "environment"
    save_json(job["output"],environment_info())
elseif job["action"] == "export"
    export_inputs()
elseif job["action"] == "check"
    check_model()
elseif job["action"] == "audit"
    audit_fixed()
elseif job["action"] == "baseline"
    include(joinpath(@__DIR__, "baseline.jl"))
    run_baseline()
else
    error("Unknown author action")
end
