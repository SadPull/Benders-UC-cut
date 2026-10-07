function [candidates,info] = llm(b,best,lower,o,seconds)
%LLM Optional synchronous Responses API advisor; candidates are untrusted.
candidates=[]; info=struct('status','FAILED','message','','accepted',0);
context=struct('hours',b.times,'unit_count',b.d.n_units,'current_lower_bound',lower, ...
    'current_cost',best.cost,'u',best.point(b.u),'v',best.point(b.v),'w',best.point(b.w), ...
    'pmin',b.d.pmin,'pmax',b.d.pmax,'pstart',b.d.pstart,'pshut',b.d.pshut, ...
    'min_up',b.d.min_up,'min_down',b.d.min_down,'ramp_up',b.d.ramp_up,'ramp_down',b.d.ramp_down, ...
    'priced_binary_cost',b.obj(b.binary),'scenario_loads',[]);
for k=1:numel(b.scenarios), context.scenario_loads(k,:)=b.scenarios(k).load(b.times); end
system=['Propose up to 3 alternative binary schedules for this stochastic unit commitment block. ' ...
    'Return ONLY JSON: {"candidates":[{"u":[...],"v":[...],"w":[...]}]}. ' ...
    'Each matrix has unit_count rows and one column per listed hour; entries must be 0 or 1. ' ...
    'u is commitment, v startup, w shutdown. Local costs include Lagrangian prices. ' ...
    'At a block boundary do not invent a reset to the original initial condition. ' ...
    'All proposals will be checked against the actual constraints and dispatch QP.'];
body=struct('model',o.llm_model,'instructions',system,'input',jsonencode(context), ...
    'stream',false,'store',false);
try
    settings=weboptions('MediaType','application/json','ContentType','json', ...
        'HeaderFields',{'Authorization',['Bearer ' o.llm_api_key]}, ...
        'Timeout',max(1,min(seconds,o.llm_timeout)));
    response=webwrite(o.llm_url,body,settings);
    assert(strcmp(response.status,'completed'),'suc:LlmResponse','Response is not completed.');
    output=response.output;
    if ~iscell(output), output=num2cell(output); end
    content='';
    for k=1:numel(output)
        item=output{k};
        if ~strcmp(item.type,'message'), continue; end
        parts=item.content;
        if ~iscell(parts), parts=num2cell(parts); end
        for j=1:numel(parts)
            part=parts{j};
            if strcmp(part.type,'output_text'), content=[content part.text]; end %#ok<AGROW>
        end
    end
    content=strtrim(content);
    assert(~isempty(content),'suc:LlmResponse','Response contains no output text.');
    content=regexprep(content,'^```(?:json)?\s*|\s*```$','');
    raw=jsondecode(content);
    assert(isstruct(raw) && isfield(raw,'candidates'),'Invalid candidate JSON.');
    for k=1:min(3,numel(raw.candidates))
        row=raw.candidates(k); valid=true; x=zeros(numel(b.binary),1);
        for name={'u','v','w'}
            f=name{1};
            if ~isfield(row,f), valid=false; break; end
            value=row.(f);
            if ~isequal(size(value),size(b.(f))) || ~isnumeric(value) || any(~isfinite(value),'all') ...
                    || any(value~=0 & value~=1,'all'), valid=false; break; end
            [~,where]=ismember(b.(f),b.binary); x(where)=value;
        end
        if valid, candidates(:,end+1)=x; end %#ok<AGROW>
    end
    candidates=unique(candidates','rows','stable')';
    info.status='OK'; info.accepted=size(candidates,2);
catch exception
    % Do not persist request headers, raw responses or credentials.
    info.message=exception.identifier;
end
end
