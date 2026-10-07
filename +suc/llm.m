function [candidates,info] = llm(b,best,lower,o,seconds,feedback)
%LLM Optional synchronous Responses API advisor; candidates are untrusted.
if nargin < 6, feedback=struct(); end
[body,info]=suc.llm_request(b,best,lower,o,feedback); candidates=[];
attempt_folder='';
try
    if isfield(o,'llm_budget_dir') && ~isempty(o.llm_budget_dir)
        limit=6;
        if isfield(o,'llm_global_max_calls'), limit=o.llm_global_max_calls; end
        [attempt_folder,budget_cleanup]=claim(o.llm_budget_dir,limit); %#ok<NASGU>
        write_json(fullfile(attempt_folder,'request.json'),body);
    end
    settings=weboptions('MediaType','application/json','ContentType','json', ...
        'HeaderFields',{'Authorization',['Bearer ' o.llm_api_key]}, ...
        'Timeout',max(1,min(seconds,o.llm_timeout)));
    started=tic;
    response=webwrite(o.llm_url,body,settings);
    elapsed=toc(started);
    if ~isempty(attempt_folder), write_json(fullfile(attempt_folder,'response.json'),response); end
    [candidates,parsed]=suc.llm_parse(response,b);
    parsed.context_seconds=info.context_seconds; parsed.http_seconds=elapsed; info=parsed;
catch exception
    % Persist identifiers only: transport messages can contain credentials.
    info.status='FAILED'; info.message=exception.identifier;
    if exist('started','var'), info.http_seconds=toc(started); end
end
if ~isempty(attempt_folder)
    % No configuration, request headers or raw exception messages enter artifacts.
    write_json(fullfile(attempt_folder,'info.json'),info);
    path=fullfile(o.llm_budget_dir,'call-ledger.json');
    entries=jsondecode(fileread(path,'Encoding','UTF-8'));
    entries(end).status=info.status; write_json(path,entries);
end
end

function [folder,cleanup]=claim(root,limit)
% An occupied slot remains consumed even if the process dies before sending.
validateattributes(limit,{'numeric'},{'scalar','integer','positive','finite'});
if ~isfolder(root), mkdir(root); end
file=java.io.RandomAccessFile(fullfile(root,'call-ledger.lock'),'rw');
channel=file.getChannel();
try
    lock=channel.tryLock();
catch
    channel.close(); file.close();
    error('suc:LlmBudgetLock','Another request owns the global call lock.');
end
if isempty(lock)
    channel.close(); file.close();
    error('suc:LlmBudgetLock','Another request owns the global call lock.');
end
cleanup=onCleanup(@()release(lock,channel,file));
path=fullfile(root,'call-ledger.json');
entries=struct('attempt_id',{},'directory',{},'started_utc',{},'status',{});
if isfile(path)
    entries=jsondecode(fileread(path,'Encoding','UTF-8'));
    assert(isempty(entries) || (isstruct(entries) && all(isfield(entries, ...
        {'attempt_id','directory','started_utc','status'}))),'suc:LlmBudget','Invalid global ledger.');
end
assert(numel(entries)<limit,'suc:LlmBudget','Global LLM call budget exhausted.');
id=char(java.util.UUID.randomUUID()); folder=fullfile(root,id);
entry=struct('attempt_id',id,'directory',folder, ...
    'started_utc',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss.SSS''Z''')), ...
    'status','CLAIMED');
if isempty(entries), entries=entry; else, entries(end+1)=entry; end
write_json(path,entries);
mkdir(folder);
end

function release(lock,channel,file)
lock.release(); channel.close(); file.close();
end

function write_json(path,value)
temporary=[path '.tmp']; fid=fopen(temporary,'w','n','UTF-8');
assert(fid>=0,'suc:LlmArtifact','Cannot open LLM artifact.');
cleanup=onCleanup(@()fclose(fid)); fprintf(fid,'%s',jsonencode(value)); clear cleanup;
movefile(temporary,path,'f');
end
