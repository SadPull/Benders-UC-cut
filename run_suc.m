function result = run_suc(options)
%RUN_SUC Time-block Benders / native MATLAB SA for stochastic UC.
% Edit llm_config.m once to configure the optional Responses API advisor.
% result = run_suc(struct('max_outer',2,'max_inner',5,'time_limit',120));
if nargin == 0, options = struct(); end
root = fileparts(mfilename('fullpath'));
defaults = struct('data_file',fullfile(root,'data','units.json'), ...
    'scenario_file',fullfile(root,'data','stochastic.json'), ...
    'block_hours',3,'max_outer',20,'max_inner',20,'tolerance',1e-4, ...
    'time_limit',600,'certify_seconds',30,'seed',7,'verbose',true, ...
    'theta_step',1,'penalty',1e6,'reserve_scale',100, ...
    'sa_reads',4,'sa_sweeps',2,'sa_temperature',20,'sa_cooling',0.9, ...
    'sa_cutoff',0.1,'capture_stagnation',false,'fusion_mode','C','llm_min_remaining',0, ...
    'llm_budget_dir','','llm_global_max_calls',6);
llm = llm_config();
for name = fieldnames(llm)'
    defaults.(name{1}) = llm.(name{1});
end
names = fieldnames(options);
for k = 1:numel(names)
    assert(isfield(defaults,names{k}),'suc:Options','Unknown option: %s',names{k});
    defaults.(names{k}) = options.(names{k});
end
o = defaults;
assert(ismember(o.fusion_mode,{'C','D'}),'suc:Options','fusion_mode must be C or D.');
validateattributes(o.llm_min_remaining,{'numeric'},{'scalar','nonnegative','finite'});
validateattributes(o.capture_stagnation,{'logical'},{'scalar'});
for name = {'block_hours','max_outer','max_inner','sa_reads','sa_sweeps','llm_patience'}
    validateattributes(o.(name{1}),{'numeric'},{'scalar','integer','positive','finite'},'',name{1});
end
for name = {'time_limit','certify_seconds','theta_step','penalty','reserve_scale','sa_temperature','sa_cutoff','llm_timeout'}
    validateattributes(o.(name{1}),{'numeric'},{'scalar','positive','finite'},'',name{1});
end
validateattributes(o.tolerance,{'numeric'},{'scalar','nonnegative','finite'});
validateattributes(o.seed,{'numeric'},{'scalar','integer','nonnegative','finite'});
validateattributes(o.llm_max_calls,{'numeric'},{'scalar','integer','nonnegative','finite'});
assert(o.sa_cooling > 0 && o.sa_cooling < 1 && o.sa_cutoff < o.sa_temperature, ...
    'suc:Options','Require 0 < sa_cooling < 1 and sa_cutoff < sa_temperature.');
assert(exist('gurobi','file') ~= 0,'suc:Dependency', ...
    'Add the Gurobi MATLAB interface to your path and activate its license.');
if o.llm_enabled
    assert(~isempty(o.llm_url) && ~isempty(o.llm_model) && ~isempty(o.llm_api_key), ...
        'suc:Options','Fill llm_url, llm_model and llm_api_key in llm_config.m before enabling LLM.');
end
[d,s] = suc.load_data(o.data_file,o.scenario_file);
state = rng; cleanup = onCleanup(@() rng(state));
rng(o.seed,'twister');
result = suc.solve(d,s,o);
end
