function summary = stagnation_online_report(directory)
%STAGNATION_ONLINE_REPORT Offline full-run results and within-block attribution.
parent=fileparts(directory); modes='AC';
[screen,ok]=read_json(fullfile(parent,'summary.json'));
if ok && strcmp(field(screen,'selected_mode',''),'D'), modes='AD'; end
if ~ok && ~isempty(dir(fullfile(directory,'seed-*','D-online.mat'))), modes='AD'; end
[ledger,has_ledger]=read_json(fullfile(directory,'online-calls','call-ledger.json'));
reference=struct('status','ABSENT','objective',NaN,'lower',NaN);
[raw,ok]=read_mat(fullfile(parent,'full-reference.mat'),'reference');
if ok
    reference.status=field(raw,'status','UNKNOWN');
    reference.objective=field(raw,'objval',NaN); reference.lower=field(raw,'lower',NaN);
end
runs=[];
for seed=[107 117 127]
    for mode=modes
        row=struct('seed',seed,'mode',mode,'artifact_status','PENDING','status','PENDING', ...
            'objective',NaN,'lower_bound',NaN,'relative_gap',NaN,'reference_difference',NaN, ...
            'reference_difference_fraction',NaN,'elapsed_seconds',NaN,'llm_calls',NaN, ...
            'first_full_improvement_seconds',NaN,'first_full_improvement_status','NOT_RECORDED', ...
            'last_outer',NaN,'last_block',NaN,'last_inner',NaN,'validation_status','UNKNOWN', ...
            'max_residual_mw',NaN,'cost_residual',NaN, ...
            'attribution_coverage','COMPLETED_HISTORY_ONLY','may_have_unrecorded_tail',false, ...
            'attribution_status','UNKNOWN','attribution_reason','RESULT_MISSING','blocks',[]);
        [r,ok]=read_mat(fullfile(directory,sprintf('seed-%d',seed),[mode '-online.mat']),'result');
        if ok
            row.artifact_status='COMPLETE'; row.status=field(r,'status','UNKNOWN');
            row.may_have_unrecorded_tail=strcmp(row.status,'TIME_LIMIT') || contains(row.status,'FAILED');
            row.objective=field(r,'objective',NaN); row.lower_bound=field(r,'lower_bound',NaN);
            row.relative_gap=field(r,'relative_gap',NaN); row.elapsed_seconds=field(r,'elapsed_seconds',NaN);
            row.llm_calls=field(r,'llm_calls',NaN);
            history=field(r,'history',[]);
            if ~isempty(history)
                row.last_outer=field(history(end),'outer',NaN);
                row.last_block=field(history(end),'block',NaN); row.last_inner=field(history(end),'inner',NaN);
            end
            validation=field(r,'validation',[]);
            if isstruct(validation) && isfield(validation,'feasible')
                row.validation_status='FAILED';
                if validation.feasible, row.validation_status='VALIDATED'; end
                row.max_residual_mw=field(validation,'max_residual_mw',NaN);
                row.cost_residual=field(validation,'cost_residual',NaN);
            end
            row.reference_difference=row.objective-reference.objective;
            row.reference_difference_fraction=row.reference_difference/max(1,abs(reference.objective));
            [task,has_task]=read_json(fullfile(directory,'tasks',sprintf('online-%d-%s.done.json',seed,mode)));
            [row.blocks,row.attribution_status,row.attribution_reason]=attribute(r,task,has_task,ledger,has_ledger);
        end
        runs=[runs row]; %#ok<AGROW>
    end
end
pairs=[];
for k=1:2:numel(runs)
    a=runs(k); c=runs(k+1);
    pair=struct('seed',a.seed,'fusion_mode',c.mode,'status','PENDING', ...
        'control_objective',a.objective,'fusion_objective',c.objective, ...
        'fusion_improvement',NaN,'fusion_improvement_fraction',NaN);
    if strcmp(a.artifact_status,'COMPLETE') && strcmp(c.artifact_status,'COMPLETE')
        pair.status='UNVALIDATED';
        if isfinite(a.objective) && isfinite(c.objective)
            pair.fusion_improvement=a.objective-c.objective;
            pair.fusion_improvement_fraction=pair.fusion_improvement/max(1,abs(a.objective));
            if strcmp(a.validation_status,'VALIDATED') && strcmp(c.validation_status,'VALIDATED')
                pair.status='VALIDATED';
            end
        end
    end
    pairs=[pairs pair]; %#ok<AGROW>
end
screening=same_screening_state(parent);
summary=struct('status','PENDING','modes',modes,'time_limit_seconds',1800,'max_inner',200, ...
    'reference',reference,'screening_context',screening,'runs',runs,'pairs',pairs);
if all(strcmp({runs.artifact_status},'COMPLETE')), summary.status='COMPLETE'; end
write_text(fullfile(directory,'online-analysis.json'),jsonencode(summary,'PrettyPrint',true));
lines={'# SA 与融合方法完整运行对照','', ...
    '每次运行同时受 1800 秒总时间与每块 200 次内层迭代上限约束，不保证跑满 1800 秒。块内尚未收敛即达到迭代上限时可返回 INNER_NOT_CONVERGED；按实际耗时与停止状态解释结果。', ...
    '全天目标、下界和间隙来自各完整运行。参考差为目标减参考可行目标；参考未证最优时不能称最优差。', ...
    '首次全天改善和首次达标时刻未记录（NOT_RECORDED）。总耗时不能代替首次时间；若已发生改善，总耗时只给出接受该改善时刻的上界。', ...
    '归因按外层轮次与块分别计算，负成本使用直接差值；不同块或乘子下的定价成本不相加。缺失或不匹配标 UNKNOWN，不能解释成零收益。', ...
    '归因仅覆盖 history 已记录的完整批次；MATCHED 只表示这些记录和请求匹配，不证明覆盖全部收益。TIME_LIMIT 或异常可能中断末批，已接受但尚未写入 history 的候选改善无法据此分解。全天最终目标仍以 result 为准。', ...
    '调用所在批次：直接改善=请求成本−该批上界；调用前 SA 改善=同块前批上界−请求成本。首个调用后的其他批次用同块相邻上界差。该分解是成本来源记录，不是因果证明。','', ...
    '| 种子 | 方法 | 状态 | 全天目标 | 下界 | 间隙% | 参考差 | 秒 | 调用数 | 归因 |', ...
    '|---|---|---|---:|---:|---:|---:|---:|---:|---|'};
for k=1:numel(runs)
    r=runs(k);
    lines{end+1}=sprintf('| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |', ...
        r.seed,r.mode,r.status,num(r.objective),num(r.lower_bound),num(100*r.relative_gap), ...
        num(r.reference_difference),num(r.elapsed_seconds),num(r.llm_calls),r.attribution_status); %#ok<AGROW>
end
lines=[lines {'',sprintf('参考状态：%s；可行目标 %s；认证下界 %s。', ...
    reference.status,num(reference.objective),num(reference.lower))}];
if screening.same_block_iteration_and_cost
    lines{end+1}=sprintf('首批三个检查点均为块 %d、第 %d 次迭代、相同 incumbent 成本 %s；这是同一早期停滞状态附近的随机种子重复，不代表三种不同陷阱。', ...
        screening.block,screening.inner,num(screening.upper));
end
lines=[lines {'','配对全天改善使用 A 的成本减融合成本，正值表示融合更好。', ...
    '| 种子 | 融合方法 | 校验 | 全天降本 | 相对降本% |','|---|---|---|---:|---:|'}];
for k=1:numel(pairs)
    p=pairs(k);
    lines{end+1}=sprintf('| %d | %s | %s | %s | %s |', ...
        p.seed,p.fusion_mode,p.status,num(p.fusion_improvement),num(100*p.fusion_improvement_fraction)); %#ok<AGROW>
end
for k=1:numel(runs)
    r=runs(k);
    lines{end+1}=sprintf('\n种子 %d / %s：归因 %s（%s）。', ...
        r.seed,r.mode,r.attribution_status,r.attribution_reason); %#ok<AGROW>
    lines{end+1}=sprintf('最后已记录位置：outer=%s、block=%s、inner=%s；全天校验 %s，最大物理残差 %s MW，成本残差 %s。未记录的中断位置不作推断。', ...
        num(r.last_outer),num(r.last_block),num(r.last_inner),r.validation_status, ...
        num(r.max_residual_mw),num(r.cost_residual)); %#ok<AGROW>
    if r.may_have_unrecorded_tail
        lines{end+1}='本次为时间截止或异常结束，最后已记录批次之后可能存在未记录的末批；其收益不能按零计。'; %#ok<AGROW>
    end
    for j=1:numel(r.blocks)
        b=r.blocks(j);
        lines{end+1}=sprintf('外层 %d、块 %d：%s；LLM 直接改善 %s；调用批次前段 SA 改善 %s；首调用后非调用批次 SA 改善 %s；首调用后 SA 总改善 %s。', ...
            b.outer,b.block,b.status,num(b.llm_direct_improvement),num(b.call_iteration_sa_improvement), ...
            num(b.later_noncall_sa_improvement),num(b.sa_after_first_call_improvement)); %#ok<AGROW>
    end
end
write_text(fullfile(directory,'online-report.md'),strjoin(lines,newline));
end

function context=same_screening_state(parent)
context=struct('same_block_iteration_and_cost',false,'block',NaN,'inner',NaN,'upper',NaN);
coordinates=[];
for seed=[7 17 27]
    [capture,ok]=read_json(fullfile(parent,sprintf('seed-%d',seed),'capture-status.json'));
    if ~ok || ~strcmp(field(capture,'status',''),'STAGNATION_CAPTURED'), return; end
    coordinates(end+1,:)=[field(capture,'block',NaN),field(capture,'inner',NaN),field(capture,'upper',NaN)]; %#ok<AGROW>
end
if all(isfinite(coordinates),'all') && all(coordinates==coordinates(1,:),'all')
    context.same_block_iteration_and_cost=true;
    context.block=coordinates(1,1); context.inner=coordinates(1,2); context.upper=coordinates(1,3);
end
end

function [blocks,status,reason]=attribute(r,task,has_task,ledger,has_ledger)
blocks=[]; status='UNKNOWN'; reason='CALL_MATCH_UNKNOWN';
h=field(r,'history',[]); calls=field(r,'llm_history',[]); count=field(r,'llm_calls',NaN);
if count==0 && isempty(calls), status='NOT_APPLICABLE'; reason='NO_LLM_CALLS'; return; end
if isempty(h), reason='HISTORY_MISSING'; return; end
if ~isstruct(h) || ~all(isfield(h,{'outer','block','inner','upper'}))
    reason='HISTORY_INVALID'; return;
end
keys=unique([[h.outer]' [h.block]'],'rows','stable');
for k=1:size(keys,1)
    blocks=[blocks struct('outer',keys(k,1),'block',keys(k,2),'status','UNKNOWN', ...
        'llm_direct_improvement',NaN,'call_iteration_sa_improvement',NaN, ...
        'later_noncall_sa_improvement',NaN,'sa_after_first_call_improvement',NaN,'calls',[])]; %#ok<AGROW>
end
if ~has_task, reason='TASK_WINDOW_MISSING'; return; end
if ~has_ledger, reason='LEDGER_MISSING'; return; end
start=utc_seconds(field(task,'started_utc','')); stop=utc_seconds(field(task,'ended_utc',''));
if ~isfinite(start) || ~isfinite(stop) || stop<start, reason='TASK_WINDOW_INVALID'; return; end
times=arrayfun(@(e)utc_seconds(field(e,'started_utc','')),ledger);
if any(~isfinite(times)), reason='LEDGER_TIME_INVALID'; return; end
indices=find(times>=start & times<=stop); [~,order]=sort(times(indices)); indices=indices(order);
if numel(indices)~=numel(calls) || count~=numel(calls), reason='CALL_COUNT_MISMATCH'; return; end
if ~isstruct(calls) || ~all(isfield(calls,{'outer','block','inner'}))
    reason='CALL_HISTORY_INVALID'; return;
end
if numel(unique(times(indices)))~=numel(indices), reason='CALL_TIME_AMBIGUOUS'; return; end
request_cost=nan(1,numel(calls)); match=nan(1,numel(calls));
for k=1:numel(calls)
    entry=ledger(indices(k));
    [body,ok]=read_json(fullfile(field(entry,'directory',''),'request.json'));
    if ~ok, reason='REQUEST_MISSING'; return; end
    try, context=jsondecode(body.input); catch, reason='REQUEST_INVALID'; return; end
    request_cost(k)=field(context,'current_cost',NaN);
    found=find([h.outer]==calls(k).outer & [h.block]==calls(k).block & [h.inner]==calls(k).inner);
    if numel(found)~=1 || ~isfinite(request_cost(k)), reason='CALL_HISTORY_MISMATCH'; return; end
    match(k)=found;
end
if numel(unique(match))~=numel(match) || any(diff(match)<=0), reason='CALL_ORDER_MISMATCH'; return; end
for j=1:numel(blocks)
    b=blocks(j); hi=find([h.outer]==b.outer & [h.block]==b.block);
    ci=find([calls.outer]==b.outer & [calls.block]==b.block);
    if isempty(ci), b.status='NOT_APPLICABLE'; blocks(j)=b; continue; end
    direct=[]; before=[]; after=[]; records=[]; first=match(ci(1));
    for k=ci
        at=match(k); prev=hi(find(hi<at,1,'last')); sa=NaN;
        if ~isempty(prev) && h(at).inner==h(prev).inner+1, sa=h(prev).upper-request_cost(k); end
        d=request_cost(k)-h(at).upper;
        if ~isfinite(d), d=NaN; end
        if ~isfinite(sa), sa=NaN; end
        direct(end+1)=d; before(end+1)=sa; %#ok<AGROW>
        if at>first, after(end+1)=sa; end %#ok<AGROW>
        records=[records struct('inner',h(at).inner,'ledger_status',field(ledger(indices(k)),'status','UNKNOWN'), ...
            'request_cost',request_cost(k),'row_upper',h(at).upper,'row_lower',field(h(at),'lower',NaN), ...
            'llm_direct_improvement',d,'preceding_sa_improvement',sa)]; %#ok<AGROW>
    end
    later=[];
    for at=hi(hi>first)
        if ismember(at,match), continue; end
        prev=hi(find(hi<at,1,'last')); delta=NaN;
        if ~isempty(prev) && h(at).inner==h(prev).inner+1, delta=h(prev).upper-h(at).upper; end
        if ~isfinite(delta), delta=NaN; end
        later(end+1)=delta; %#ok<AGROW>
    end
    b.calls=records; b.llm_direct_improvement=sum(direct);
    b.call_iteration_sa_improvement=sum(before); b.later_noncall_sa_improvement=sum(later);
    b.sa_after_first_call_improvement=sum([after later]); b.status='MATCHED';
    if any(isnan([direct before later])), b.status='PARTIAL'; end
    blocks(j)=b;
end
status='MATCHED'; reason='WINDOW_AND_HISTORY_MATCHED';
if any(strcmp({blocks.status},'PARTIAL')), status='PARTIAL'; reason='PREVIOUS_ITERATION_UNKNOWN'; end
end

function seconds=utc_seconds(text)
seconds=NaN;
parts=regexp(char(text),'^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)Z$','tokens','once');
if numel(parts)==6, seconds=datenum(cellfun(@str2double,parts))*86400; end
end

function [data,ok]=read_mat(path,name)
data=[]; ok=false;
if ~isfile(path), return; end
try
    raw=load(path,name);
    if isfield(raw,name), data=raw.(name); ok=isstruct(data); end
catch
end
end

function [data,ok]=read_json(path)
data=[]; ok=false;
if ~isfile(path), return; end
try, data=jsondecode(fileread(path,'Encoding','UTF-8')); ok=true; catch, end
end

function value=field(s,name,fallback)
if isstruct(s) && isfield(s,name), value=s.(name); else, value=fallback; end
end

function text=num(value)
if isfinite(value), text=sprintf('%.6f',value); else, text='UNKNOWN'; end
end

function write_text(path,text)
file=fopen(path,'w','n','UTF-8'); assert(file>=0,'suc:Report','Cannot write online report.');
cleanup=onCleanup(@()fclose(file)); %#ok<NASGU>
fprintf(file,'%s\n',text);
end
