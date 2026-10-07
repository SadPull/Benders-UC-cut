function run_stagnation_experiment(action,directory,seed,mode)
%RUN_STAGNATION_EXPERIMENT One resumable, independently scheduled experiment task.
if nargin<3, seed=7; end
if nargin<4, mode=''; end
if ~isfolder(directory), mkdir(directory); end
folder=fullfile(directory,sprintf('seed-%d',seed));
if ~isfolder(folder), mkdir(folder); end
switch action
    case 'capture'
        target=fullfile(folder,'capture.mat');
        if isfile(target), return; end
        result=run_suc(struct('seed',seed,'llm_enabled',false,'capture_stagnation',true, ...
            'max_inner',200,'time_limit',1800));
        atomic_save(target,'result',result);
        if isfield(result,'checkpoint')
            write_json(fullfile(folder,'capture-status.json'),struct('status',result.status, ...
                'block',result.checkpoint.index,'inner',result.checkpoint.state.iteration, ...
                'upper',result.checkpoint.state.upper,'elapsed',result.elapsed_seconds));
        else
            write_json(fullfile(folder,'capture-status.json'),struct('status',result.status,'triggered',false));
        end
    case 'prepare'
        target=fullfile(folder,'request.json');
        if isfile(target), return; end
        cp=checkpoint(folder); if isempty(cp), return; end
        o=llm_config(); o.llm_model='gpt-6-astra';
        feedback=struct('cuts',cp.state.cuts(max(1,end-7):end), ...
            'recent',cp.state.events(max(1,end-11):end));
        [body,info]=suc.llm_request(cp.b,cp.state.best,cp.state.lower,o,feedback);
        atomic_save(fullfile(folder,'request-info.mat'),'info',info);
        write_json(target,body);
    case 'import'
        target=fullfile(folder,'response.mat');
        if isfile(target), return; end
        cp=checkpoint(folder); if isempty(cp), return; end
        loaded=load(fullfile(folder,'request-info.mat'),'info');
        http=read_json(fullfile(folder,'http-info.json'));
        protocol=0; if isfield(http,'protocol_seconds'), protocol=http.protocol_seconds; end
        shared=struct('status','FAILED','candidates',[],'info',http, ...
            'charged_seconds',loaded.info.context_seconds+http.http_seconds+protocol);
        if http.completed
            response=read_json(fullfile(folder,'response.json'));
            [shared.candidates,parsed]=suc.llm_parse(response,cp.b);
            shared.status=parsed.status;
            shared.charged_seconds=shared.charged_seconds+parsed.parse_seconds;
            shared.parse_info=parsed;
        end
        atomic_save(target,'shared',shared);
    case 'branch'
        assert(ismember(mode,{'A','B','C','D'}),'suc:Experiment','Unknown branch.');
        target=fullfile(folder,[mode '.mat']); if isfile(target), return; end
        cp=checkpoint(folder); if isempty(cp), return; end
        o=cp.o; o.time_limit=1200; o.llm_enabled=false;
        intervention=struct('mode',mode,'charged_seconds',0);
        if ismember(mode,{'C','D'})
            loaded=load(fullfile(folder,'response.mat'),'shared');
            intervention.proposals=loaded.shared.candidates;
            intervention.charged_seconds=loaded.shared.charged_seconds;
        end
        began=datetime('now','TimeZone','UTC');
        [out,history,state]=suc.search_block(cp.b,cp.seeds,o,tic,0,cp.outer,cp.index,cp.state,intervention);
        branch=struct('mode',mode,'seed',seed,'baseline_cost',cp.state.upper,'out',out, ...
            'history',history,'events',out.evaluation_events,'state',state, ...
            'started_utc',char(began),'ended_utc',char(datetime('now','TimeZone','UTC')), ...
            'full_check',struct('status','PENDING','feasible',false,'objective',Inf));
        atomic_save(target,'branch',branch);
    case 'diagnose'
        target=fullfile(folder,[mode '.mat']); if ~isfile(target), return; end
        loaded=load(target,'branch'); branch=loaded.branch;
        target=fullfile(folder,[mode '-full.mat']); if isfile(target), return; end
        cp=checkpoint(folder); began=tic;
        proposal=cp.incumbent.u;
        if ~isempty(branch.out.best), proposal(:,cp.b.times)=branch.out.best.point(cp.b.u); end
        full_check=check_full(cp.d,cp.s,proposal,began,60);
        atomic_save(target,'full_check',full_check);
    case 'reference'
        target=fullfile(folder,'block-reference.mat');
        if ~isfile(target)
            cp=checkpoint(folder);
            if ~isempty(cp)
                reference=suc.optimize(cp.b,60,true);
                reference.lower=reference.lower+cp.b.constant;
                if isfield(reference,'objval'), reference.objval=reference.objval+cp.b.constant; end
                atomic_save(target,'reference',reference);
            end
        end
        target=fullfile(directory,'full-reference.mat');
        if ~isfile(target)
            root=fileparts(mfilename('fullpath'));
            [d,s]=suc.load_data(fullfile(root,'data','units.json'),fullfile(root,'data','stochastic.json'));
            reference=suc.optimize(suc.model(d,s),600,true);
            atomic_save(target,'reference',reference);
        end
    case 'report'
        suc.stagnation_report(directory);
    case 'online'
        target=fullfile(folder,[mode '-online.mat']); if isfile(target), return; end
        options=struct('seed',seed,'time_limit',1800,'max_inner',200,'llm_enabled',~strcmp(mode,'A'), ...
            'llm_model','gpt-6-astra','llm_max_calls',2,'llm_timeout',300,'llm_min_remaining',600, ...
            'llm_budget_dir',fullfile(directory,'online-calls'),'llm_global_max_calls',6);
        if ~strcmp(mode,'A'), options.fusion_mode=mode; end
        % Credentials remain in the user's config, outside the frozen source.
        if options.llm_enabled
            config_path=getenv('SUC_LLM_CONFIG');
            assert(~isempty(config_path),'suc:Experiment','Missing external LLM configuration.');
            text=fileread(config_path);
            for name={'llm_url','llm_api_key'}
                tokens=regexp(text,['(?m)^\s*o\.' name{1} '\s*=\s*''((?:''''|[^''])*)''\s*;'],'tokens');
                assert(numel(tokens)==1,'suc:Experiment','Invalid external LLM configuration.');
                options.(name{1})=strrep(tokens{1}{1},'''''','''');
            end
        end
        result=run_suc(options); atomic_save(target,'result',result);
    case 'online-report'
        rows=[];
        for run_seed=[107 117 127]
            for arm=['A' mode]
                path=fullfile(directory,sprintf('seed-%d',run_seed),[arm '-online.mat']);
                if ~isfile(path), continue; end
                loaded=load(path,'result'); r=loaded.result;
                rows=[rows struct('seed',run_seed,'mode',arm,'status',r.status, ...
                    'objective',r.objective,'lower_bound',r.lower_bound,'gap',r.relative_gap, ...
                    'llm_calls',r.llm_calls,'elapsed_seconds',r.elapsed_seconds)];
            end
        end
        write_json(fullfile(directory,'online-summary.json'),rows);
    otherwise
        error('suc:Experiment','Unknown experiment action.');
end
end

function cp=checkpoint(folder)
loaded=load(fullfile(folder,'capture.mat'),'result'); cp=[];
if isfield(loaded.result,'checkpoint'), cp=loaded.result.checkpoint; end
end

function checked=check_full(d,s,u,started,budget)
checked=struct('status','INFEASIBLE','feasible',false,'objective',Inf,'elapsed_seconds',0);
m=suc.model(d,s); [blocks,~]=suc.partition(m,d.horizon); b=blocks{1};
x=suc.repair(d,s,u);
if ~isempty(x) && toc(started)<budget
    r=suc.dispatch(b,x,budget-toc(started));
    if r.feasible && toc(started)<budget
        z=struct('u',reshape(r.point(m.ix.u),d.n_units,d.horizon), ...
            'v',reshape(r.point(m.ix.v),d.n_units,d.horizon), ...
            'w',reshape(r.point(m.ix.w),d.n_units,d.horizon),'scenarios',[], ...
            'objective',m.obj'*r.point+sum(m.q.*r.point.^2));
        for k=1:numel(s)
            p=reshape(r.point(m.ix.p(:,:,k)),d.n_units,d.horizon);
            reserve=reshape(r.point(m.ix.r(:,:,k)),d.n_units,d.horizon);
            z.scenarios=[z.scenarios struct('name',s(k).name,'p',p,'r',reserve, ...
                'objective',sum(d.cost_a.*p.^2+d.cost_b.*p,'all'))];
        end
        checked.validation=suc.validate(d,s,z);
        if toc(started)<budget
            checked.status='VALIDATED'; checked.feasible=true; checked.objective=z.objective; checked.result=z;
        end
    else
        checked.status=r.status;
    end
end
checked.elapsed_seconds=toc(started);
if checked.elapsed_seconds>=budget, checked.status='TIME_LIMIT'; end
end

function atomic_save(path,name,value)
payload=struct(); payload.(name)=value;
temporary=[path '.tmp.mat']; save(temporary,'-struct','payload','-v7.3');
movefile(temporary,path,'f');
end

function write_json(path,value)
temporary=[path '.tmp']; fid=fopen(temporary,'w','n','UTF-8');
assert(fid>=0,'suc:ExperimentIO','Cannot open artifact.');
cleanup=onCleanup(@()fclose(fid)); fprintf(fid,'%s',jsonencode(value)); clear cleanup;
movefile(temporary,path,'f');
end

function value=read_json(path)
fid=fopen(path,'r','n','UTF-8'); assert(fid>=0,'suc:ExperimentIO','Missing JSON artifact.');
cleanup=onCleanup(@()fclose(fid)); value=jsondecode(fscanf(fid,'%c'));
end
