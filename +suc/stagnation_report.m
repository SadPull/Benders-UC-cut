function summary = stagnation_report(directory)
%STAGNATION_REPORT Summarize persisted screening artifacts without rerunning.
seeds=[7 17 27]; modes='ABCD'; records=[]; curves={};
for seed=seeds
    folder=fullfile(directory,sprintf('seed-%d',seed));
    item=struct('seed',seed,'status','PENDING','capture_status','PENDING','response_status','PENDING', ...
        'charged_seconds',NaN,'full_baseline',NaN,'common_batches',0,'branches',[], ...
        'block_reference',read_reference(fullfile(folder,'block-reference.mat')));
    [capture,ok]=read_variable(fullfile(folder,'capture.mat'),'result');
    if ~ok, records=[records item]; continue; end %#ok<AGROW>
    item.capture_status=label(value(capture,'status','UNKNOWN'));
    if ~isfield(capture,'checkpoint') || isempty(capture.checkpoint)
        item.status='NO_TRIGGER';
        if contains(item.capture_status,'FAILED'), item.status='FAILED'; end
        records=[records item]; continue; %#ok<AGROW>
    end
    cp=capture.checkpoint;
    if isfield(cp,'incumbent'), item.full_baseline=value(cp.incumbent,'objective',NaN); end
    checkpoint_inner=NaN;
    if isfield(cp,'state'), checkpoint_inner=value(cp.state,'iteration',NaN); end
    [shared,has_response]=read_variable(fullfile(folder,'response.mat'),'shared');
    if has_response
        item.response_status=label(value(shared,'status','UNKNOWN'));
        item.charged_seconds=value(shared,'charged_seconds',NaN);
    end
    histories=cell(1,4); branches=[];
    for j=1:4
        row=empty_branch(modes(j));
        [branch,ok]=read_variable(fullfile(folder,[modes(j) '.mat']),'branch');
        if ok && isfield(branch,'out') && isfield(branch.out,'upper')
            row.status='COMPLETE'; row.baseline=value(branch,'baseline_cost',NaN);
            row.upper=branch.out.upper; row.lower=value(branch.out,'lower',-Inf);
            row.elapsed_seconds=value(branch.out,'elapsed_seconds',NaN);
            row.improvement_fraction=gain(row.baseline,row.upper);
            row.relative_gap=max(0,row.upper-row.lower)/max(1,abs(row.upper));
            row.intervention_batches=value(branch.out,'intervention_batches',0);
            events=value(branch,'events',[]);
            curves(end+1,:)={seed,modes(j),0,row.baseline,NaN,'CHECKPOINT'}; %#ok<AGROW>
            for k=1:numel(events)
                e=events(k); finished=value(e,'completed_seconds',value(e,'elapsed',NaN));
                % A result that completed beyond the online deadline cannot count.
                if ~isfinite(finished) || finished>1200, continue; end
                source=label(value(e,'source','UNKNOWN'));
                delta=value(e,'improvement',0);
                if ~isfinite(delta) || delta<0, delta=0; end
                if strcmp(source,'LLM'), row.llm_direct_improvement=row.llm_direct_improvement+delta; end
                if strcmp(source,'SA'), row.sa_improvement=row.sa_improvement+delta; end
                if ismember(modes(j),'CD') && strcmp(source,'SA') && value(e,'newbest',false) && ...
                        second_read_origin(value(e,'origins',[]))
                    row.supported_second_read_improvement=row.supported_second_read_improvement+delta;
                    row.supported_second_read_newbests=row.supported_second_read_newbests+1;
                end
                if isnan(row.first_breakthrough_seconds) && gain(row.baseline,value(e,'after',Inf))>5e-4
                    row.first_breakthrough_seconds=finished;
                end
                curves(end+1,:)={seed,modes(j),finished,value(e,'after',NaN),NaN,source}; %#ok<AGROW>
            end
            curves(end+1,:)={seed,modes(j),row.elapsed_seconds,row.upper,row.lower,'FINAL'}; %#ok<AGROW>
            [f,has_full]=read_variable(fullfile(folder,[modes(j) '-full.mat']),'full_check');
            if ~has_full && isfield(branch,'full_check'), f=branch.full_check; has_full=true; end
            if has_full
                row.full_status=label(value(f,'status','UNKNOWN'));
                row.full_feasible=logical(value(f,'feasible',false));
                if row.full_feasible, row.full_objective=value(f,'objective',NaN); end
                row.full_improvement=item.full_baseline-row.full_objective;
            end
            if ismember(row.full_status,{'PENDING','RUNNING','UNKNOWN'}), row.status='PENDING'; end
            history=value(branch,'history',[]); selected=[];
            for k=1:numel(history)
                % The first resumed row finishes the already sampled checkpoint batch.
                after_checkpoint=(isfinite(checkpoint_inner) && history(k).inner>checkpoint_inner) || ...
                    (~isfinite(checkpoint_inner) && k>1);
                if after_checkpoint && isfield(history(k),'sampling') && ...
                        value(history(k).sampling,'seconds',0)>0
                    selected=[selected history(k)]; %#ok<AGROW>
                end
            end
            histories{j}=selected; row.completed_sa_batches=numel(selected);
            if ~isfinite(row.baseline) || ~isfinite(row.upper), row.status='FAILED'; end
        elseif ok
            row.status='FAILED';
        end
        branches=[branches row]; %#ok<AGROW>
    end
    item.common_batches=min([10 branches.completed_sa_batches]);
    if item.common_batches>0
        for j=1:4
            branches(j).common_batch_upper=histories{j}(item.common_batches).upper;
        end
    end
    item.branches=branches;
    if has_response && all(strcmp({branches.status},'COMPLETE'))
        item.status='COMPLETE';
    elseif any(strcmp({branches.status},'FAILED'))
        item.status='FAILED';
    end
    records=[records item]; %#ok<AGROW>
end

wins=zeros(1,2); full_wins=zeros(1,2); d_over_c=0;
notes={};
for k=1:numel(records)
    r=records(k);
    if ~strcmp(r.status,'COMPLETE'), continue; end
    b=r.branches; control=min([b(1:2).upper]);
    for j=1:2
        wins(j)=wins(j)+(gain(control,b(j+2).upper)>5e-4);
        full_wins(j)=full_wins(j)+(b(j+2).full_feasible && b(j+2).full_improvement>1e-5);
    end
    d_over_c=d_over_c+(gain(b(3).upper,b(4).upper)>5e-4);
end
qualified=wins>=2 & full_wins>=1;
summary=struct('status','PENDING','timing','PAIRED_REPLAY_WITH_SHARED_CALL_COST', ...
    'branch_budget_seconds',1200,'threshold_fraction',5e-4,'records',records, ...
    'full_reference',read_reference(fullfile(directory,'full-reference.mat')), ...
    'control_wins_C',wins(1),'control_wins_D',wins(2), ...
    'full_improvement_seeds_C',full_wins(1),'full_improvement_seeds_D',full_wins(2), ...
    'D_over_C_wins',d_over_c,'qualified_C',qualified(1),'qualified_D',qualified(2), ...
    'expand',false,'selected_mode','','decision','WAIT_FOR_ARTIFACTS');
if all(ismember({records.status},{'COMPLETE','NO_TRIGGER','FAILED'}))
    summary.status='COMPLETE'; summary.decision='SCREENING_NOT_PASSED';
    if any(strcmp({records.status},'FAILED'))
        summary.status='FAILED'; summary.decision='FAILED_ARTIFACTS';
    elseif qualified(2) && d_over_c>=2
        summary.expand=true; summary.selected_mode='D'; summary.decision='SELECT_D';
    elseif qualified(1)
        summary.expand=true; summary.selected_mode='C'; summary.decision='SELECT_C';
    elseif qualified(2)
        summary.decision='D_QUALIFIED_BUT_SELECTION_THRESHOLD_NOT_MET';
    end
end
write_text(fullfile(directory,'summary.json'),jsonencode(summary,'PrettyPrint',true));
lines={'seed,mode,charged_elapsed_seconds,upper,lower,source'};
for k=1:size(curves,1)
    lines{end+1}=sprintf('%d,%s,%.9g,%.12g,%.12g,%s',curves{k,:}); %#ok<AGROW>
end
write_text(fullfile(directory,'curves.csv'),strjoin(lines,newline));
lines={'# SA 停滞融合筛查报告','', ...
    '采用含共享调用成本的配对重放墙钟：C、D 各自扣除同一请求的实测构建、请求和解析时间，独立验证候选；每分支在线预算 1200 秒。', ...
    '三个固定种子仅作筛查，不构成统计显著性结论。SA 后续改善不能单独证明割或热启动的因果作用；C/D 配对差异用于判断附加热启动。', ...
    '共同批次数排除捕获前已经完成采样的检查点批次，最多比较之后的前 10 个共同完成 SA 批次。全天恢复是事后诊断，不计在线成绩。','', ...
    '| 种子 | 分支 | 状态 | 块最终成本 | 块改善% | LLM直接降本 | 后续SA降本 | 全天成本 | 共同批次成本 |', ...
    '|---|---|---|---:|---:|---:|---:|---:|---:|'};
for k=1:numel(records)
    r=records(k);
    if isempty(r.branches)
        lines{end+1}=sprintf('| %d | — | %s | — | — | — | — | — | — |',r.seed,r.status); %#ok<AGROW>
    else
        for j=1:4
            b=r.branches(j);
            lines{end+1}=sprintf('| %d | %s | %s | %s | %s | %s | %s | %s | %s |', ...
                r.seed,b.mode,b.status,number(b.upper),number(100*b.improvement_fraction), ...
                number(b.llm_direct_improvement),number(b.sa_improvement), ...
                number(b.full_objective),number(b.common_batch_upper)); %#ok<AGROW>
        end
    end
    notes{end+1}=sprintf('种子 %d：%s；采集 %s；响应 %s；共同调用计费 %s 秒；共同 SA 批次 %d。', ...
        r.seed,r.status,r.capture_status,r.response_status,number(r.charged_seconds),r.common_batches); %#ok<AGROW>
    if ~isempty(r.branches)
        notes{end+1}=sprintf('全天校验状态：A=%s，B=%s，C=%s，D=%s。',r.branches.full_status); %#ok<AGROW>
        notes{end+1}=sprintf('第二 read 有父候选来源支持的新最佳解降本：C=%s，D=%s；仅作来源标签，不作因果归因。', ...
            number(r.branches(3).supported_second_read_improvement), ...
            number(r.branches(4).supported_second_read_improvement)); %#ok<AGROW>
    end
    ref=r.block_reference;
    notes{end+1}=sprintf('块参考：%s；认证下界 %s；可行目标 %s。', ...
        ref.status,number(ref.lower),number(ref.objective)); %#ok<AGROW>
end
ref=summary.full_reference;
notes{end+1}=sprintf('全天参考：%s；认证下界 %s；可行目标 %s。未获得最优证明时不称局部最优；参考只用于事后比较。', ...
    ref.status,number(ref.lower),number(ref.objective));
lines=[lines {''} notes {'',sprintf('结论状态：%s；决策：%s。',summary.status,summary.decision), ...
    sprintf('超过较好 A/B 0.05%% 的种子数：C=%d，D=%d；全天降本种子数：C=%d，D=%d；D 超过 C 0.05%%：%d。',wins,full_wins,d_over_c), ...
    '仅当候选分支自身达到至少 2 个控制组胜例、至少 1 个全天改善例时合格。D 还需至少 2 个胜 C 例；否则仅可选择自身合格的 C。缺失工件为 PENDING，不宣告成功或自动补样本。'}];
write_text(fullfile(directory,'report.md'),strjoin(lines,newline));
end

function row=empty_branch(mode)
row=struct('mode',mode,'status','PENDING','baseline',NaN,'upper',NaN,'lower',NaN, ...
    'relative_gap',NaN,'elapsed_seconds',NaN,'improvement_fraction',NaN, ...
    'llm_direct_improvement',0,'sa_improvement',0,'first_breakthrough_seconds',NaN, ...
    'supported_second_read_improvement',0,'supported_second_read_newbests',0, ...
    'intervention_batches',0,'completed_sa_batches',0,'common_batch_upper',NaN, ...
    'full_status','PENDING','full_feasible',false,'full_objective',NaN,'full_improvement',NaN);
end

function yes=second_read_origin(origins)
yes=false;
for k=1:numel(origins)
    origin=origins(k);
    if strcmp(value(origin,'kind',''),'sa') && value(origin,'read',0)==2 && ...
            ~isempty(value(origin,'parent_id',''))
        yes=true; return;
    end
end
end

function ref=read_reference(path)
ref=struct('available',false,'status','ABSENT','lower',NaN,'objective',NaN);
[raw,ok]=read_variable(path,'reference');
if ~ok, return; end
ref.available=true; ref.status=label(value(raw,'status','UNKNOWN'));
ref.lower=value(raw,'lower',NaN); ref.objective=value(raw,'objval',NaN);
end

function [data,ok]=read_variable(path,name)
data=[]; ok=false;
if ~isfile(path), return; end
try
    loaded=load(path,name);
    if isfield(loaded,name), data=loaded.(name); ok=isstruct(data); end
catch
    % Partial files are pending; never copy exception text or credentials.
end
end

function out=value(s,name,fallback)
if isstruct(s) && isfield(s,name), out=s.(name); else, out=fallback; end
end

function out=gain(before,after)
out=(before-after)/max(1,abs(before));
end

function out=label(in)
out='UNKNOWN';
if (ischar(in) || isstring(in)) && ~isempty(regexp(char(in),'^[A-Za-z0-9_]+$','once'))
    out=upper(char(in));
end
end

function out=number(in)
if isfinite(in), out=sprintf('%.6f',in); else, out='—'; end
end

function write_text(path,content)
file=fopen(path,'w','n','UTF-8'); assert(file>=0,'suc:Report','Cannot write report artifact.');
cleanup=onCleanup(@()fclose(file)); %#ok<NASGU>
fprintf(file,'%s\n',content);
end
