function [d,s] = load_data(unit_file,scenario_file)
%LOAD_DATA Validate original JSON data; keep MW and cost precision.
d = jsondecode(fileread(unit_file));
validateattributes(d.n_units,{'numeric'},{'scalar','integer','positive','finite'});
validateattributes(d.horizon,{'numeric'},{'scalar','integer','positive','finite'});
fields = {'pmax','pmin','pstart','pshut','min_up','min_down','ramp_up','ramp_down', ...
    'initial_duration','cost_a','cost_b','cost_c','startup_cost','shutdown_cost'};
for k = 1:numel(fields)
    f = fields{k};
    validateattributes(d.(f),{'numeric'},{'vector','numel',d.n_units,'finite'},'',f);
    d.(f) = d.(f)(:);
end
validateattributes(d.base_load,{'numeric'},{'vector','numel',d.horizon,'finite','nonnegative'});
d.base_load = d.base_load(:)';
for f = setdiff(fields,{'initial_duration'})
    assert(all(d.(f{1}) >= 0),'suc:Data','Negative values in %s.',f{1});
end
assert(all(d.pmin <= d.pmax) && all(d.pstart <= d.pmax) && all(d.pshut <= d.pmax), ...
    'suc:Data','Output and startup/shutdown limits must not exceed Pmax.');
for f = {'min_up','min_down'}
    validateattributes(d.(f{1}),{'numeric'},{'integer','positive'});
end
validateattributes(d.initial_duration,{'numeric'},{'integer'});
d.initial_on = double(d.initial_duration > 0);
d.initial_duration = abs(d.initial_duration);
if ~isfield(d,'initial_output'), d.initial_output = d.pmin .* d.initial_on; end
validateattributes(d.initial_output,{'numeric'},{'vector','numel',d.n_units,'finite','nonnegative'});
d.initial_output = d.initial_output(:);
assert(all(d.initial_output >= d.pmin.*d.initial_on & d.initial_output <= d.pmax.*d.initial_on), ...
    'suc:Data','Initial output is inconsistent with initial on/off state.');
raw = jsondecode(fileread(scenario_file));
d.reserve_fraction = raw.reserve_fraction;
validateattributes(d.reserve_fraction,{'numeric'},{'scalar','finite','nonnegative'});
assert(isstruct(raw.scenarios) && ~isempty(raw.scenarios),'suc:Data','Scenarios must be nonempty.');
s = struct('name',{},'probability',{},'load',{});
for k = 1:numel(raw.scenarios)
    r = raw.scenarios(k);
    assert(isfield(r,'name') && ischar(r.name) && ~isempty(r.name),'suc:Data','Scenario name required.');
    validateattributes(r.probability,{'numeric'},{'scalar','positive','finite'});
    has_factor = isfield(r,'load_factor') && ~isempty(r.load_factor);
    has_load = isfield(r,'load') && ~isempty(r.load);
    assert(xor(has_factor,has_load),'suc:Data','Specify exactly one of load and load_factor.');
    if has_factor
        validateattributes(r.load_factor,{'numeric'},{'scalar','nonnegative','finite'});
        demand = d.base_load * r.load_factor;
    else
        demand = r.load(:)';
    end
    validateattributes(demand,{'numeric'},{'vector','numel',d.horizon,'nonnegative','finite'});
    s(k) = struct('name',r.name,'probability',r.probability,'load',demand);
end
assert(numel(unique({s.name})) == numel(s),'suc:Data','Duplicate scenario names.');
assert(abs(sum([s.probability])-1) <= 1e-9,'suc:Data','Scenario probabilities must sum to one.');
assert(max([s.load])*(1+d.reserve_fraction) <= sum(d.pmax)+1e-6, ...
    'suc:Data','Load plus reserve exceeds installed capacity.');
end
