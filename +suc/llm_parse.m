function [candidates,info] = llm_parse(response,b)
%LLM_PARSE Validate syntax only; physical feasibility requires dispatch.
started=tic; candidates=[];
info=struct('status','FAILED','message','','accepted',0,'format_valid',0, ...
    'feasible',NaN,'context_seconds',0,'parse_seconds',0,'http_seconds',0);
try
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
    assert(isstruct(raw) && isfield(raw,'candidates'),'suc:LlmResponse','Invalid candidate JSON.');
    rows=raw.candidates;
    if ~iscell(rows), rows=num2cell(rows); end
    for k=1:min(3,numel(rows))
        row=rows{k}; valid=isstruct(row); x=zeros(numel(b.binary),1);
        for name={'u','v','w'}
            f=name{1};
            if ~valid || ~isfield(row,f), valid=false; break; end
            value=row.(f);
            if ~isequal(size(value),size(b.(f))) || ~isnumeric(value) || any(~isfinite(value),'all') ...
                    || any(value~=0 & value~=1,'all'), valid=false; break; end
            [~,where]=ismember(b.(f),b.binary); x(where)=value;
        end
        if valid, candidates(:,end+1)=x; end %#ok<AGROW>
    end
    info.format_valid=size(candidates,2);
    candidates=unique(candidates','rows','stable')';
    info.status='OK'; info.accepted=size(candidates,2);
catch exception
    info.message=exception.identifier;
end
info.parse_seconds=toc(started);
end
