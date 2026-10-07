param(
    [Parameter(Mandatory=$true)][string]$Directory,
    [string]$Matlab = 'C:\Program Files\MATLAB\R2025b\bin\matlab.exe',
    [double]$MinFreeGiB = 4,
    [string]$HttpWorker = ''
)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$Directory=[IO.Path]::GetFullPath($Directory)
[IO.Directory]::CreateDirectory($Directory) | Out-Null
$utf8=New-Object Text.UTF8Encoding($false)
$lock=$null
function Save-Json($path,$value) {
    [IO.File]::WriteAllText($path+'.tmp',(ConvertTo-Json -InputObject $value -Depth 40),$utf8)
    Move-Item -LiteralPath ($path+'.tmp') -Destination $path -Force
}
function Hash($path) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash }
function Free-GiB { (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB }
function Run-Task($action,$seed,$mode,$artifact) {
    $id="$action-$seed-$mode"; $done=Join-Path $Directory "tasks/$id.done.json"
    $startedMarker=Join-Path $Directory "tasks/$id.started.json"
    if(Test-Path -LiteralPath $done) {
        $prior=Get-Content -LiteralPath $done -Raw -Encoding UTF8 | ConvertFrom-Json
        if((Test-Path -LiteralPath $artifact) -and (Hash $artifact) -eq $prior.sha256) { return }
        throw "Completed artifact failed integrity check: $id"
    }
    $success=Join-Path $Directory "tasks/$id.success"
    if((Test-Path -LiteralPath $success) -and (Test-Path -LiteralPath $artifact) -and (Test-Path -LiteralPath $startedMarker)) {
        $startedRecord=Get-Content -LiteralPath $startedMarker -Raw -Encoding UTF8 | ConvertFrom-Json
        if((Get-Item -LiteralPath $success).LastWriteTimeUtc -ge ([DateTime]$startedRecord.started_utc).ToUniversalTime() -and
            (Get-Content -LiteralPath $success -Raw) -eq 'completed') {
            Save-Json $done @{id=$id;status='COMPLETE';sha256=Hash $artifact;recovery_note='Task finished after controller exited; MATLAB return marker verified.'}
            return
        }
    }
    if($action -eq 'online' -and (Test-Path -LiteralPath $startedMarker)) {
        throw "Interrupted online task cannot be automatically rerun with new model calls: $id"
    }
    $free=Free-GiB
    if($free -lt $MinFreeGiB) { throw "Resource gate: $([math]::Round($free,2)) GiB free; require $MinFreeGiB GiB. Resume later." }
    # An artifact without its completion marker belongs to an interrupted task.
    if(Test-Path -LiteralPath $artifact) {
        Move-Item -LiteralPath $artifact -Destination ($artifact+'.interrupted-'+[DateTime]::UtcNow.ToString('yyyyMMddHHmmssfff'))
    }
    $escapedDir=$Directory.Replace("'","''").Replace('\','/')
    if(Test-Path -LiteralPath $success) { Remove-Item -LiteralPath $success }
    $escapedSuccess=$success.Replace("'","''").Replace('\','/')
    $expression="run_stagnation_experiment('$action','$escapedDir',$seed,'$mode'); fid=fopen('$escapedSuccess','w'); fprintf(fid,'completed'); fclose(fid);"
    $stdout=Join-Path $Directory "logs/$id.stdout.log"; $stderr=Join-Path $Directory "logs/$id.stderr.log"
    $record=[ordered]@{id=$id;status='RUNNING';started_utc=[DateTime]::UtcNow.ToString('o');free_gib=$free;pid=$null}
    Save-Json (Join-Path $Directory 'current-task.json') $record
    Save-Json $startedMarker $record
    $process=Start-Process -FilePath $Matlab -ArgumentList "-wait -singleCompThread -batch `"$expression`"" `
        -WorkingDirectory $snapshot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $null=$process.Handle
    $record.pid=$process.Id; $record.process_started_utc=$process.StartTime.ToUniversalTime().ToString('o')
    Save-Json (Join-Path $Directory 'current-task.json') $record
    $process.WaitForExit(); $record.exit_code=$process.ExitCode
    if(($null -ne $process.ExitCode -and $process.ExitCode -ne 0) -or -not (Test-Path -LiteralPath $artifact) -or -not (Test-Path -LiteralPath $success)) {
        $record.status='FAILED'; Save-Json (Join-Path $Directory 'current-task.json') $record
        throw "Numerical task failed: $id. Inspect its saved logs; no automatic retry."
    }
    $record.status='COMPLETE'; $record.ended_utc=[DateTime]::UtcNow.ToString('o'); $record.sha256=Hash $artifact
    Save-Json $done $record; Save-Json (Join-Path $Directory 'current-task.json') $record
}
try {
    $lock=[IO.File]::Open((Join-Path $Directory 'runner.lock'),'OpenOrCreate','ReadWrite','None')
    foreach($currentPath in @((Join-Path $Directory 'current-task.json'),(Join-Path $Directory 'phase2/current-task.json'))) {
        if(Test-Path -LiteralPath $currentPath) {
            $current=Get-Content -LiteralPath $currentPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if($current.status -eq 'RUNNING' -and $current.pid) {
                $active=Get-Process -Id $current.pid -ErrorAction SilentlyContinue
                if($active -and $active.StartTime.ToUniversalTime().Ticks -eq ([DateTime]$current.process_started_utc).ToUniversalTime().Ticks) {
                    throw 'A previously controlled MATLAB task is still running; do not start a second numerical worker.'
                }
            }
        }
    }
    foreach($name in @('tasks','logs')) { [IO.Directory]::CreateDirectory((Join-Path $Directory $name)) | Out-Null }
    $snapshot=Join-Path $Directory 'snapshot'; $manifestPath=Join-Path $Directory 'manifest.json'
    if(-not (Test-Path -LiteralPath $manifestPath)) {
        if(Test-Path -LiteralPath $snapshot) { throw 'Incomplete source freeze exists; inspect before resuming.' }
        [IO.Directory]::CreateDirectory($snapshot) | Out-Null
        foreach($name in @('+suc','data','tests','scripts')) { Copy-Item -LiteralPath (Join-Path $repo $name) -Destination $snapshot -Recurse }
        Get-ChildItem -LiteralPath $repo -Filter '*.m' | Where-Object Name -ne 'llm_config.m' | Copy-Item -Destination $snapshot
        $safeConfig="function o=llm_config`no=struct('llm_enabled',false,'llm_url','','llm_api_key','','llm_model','gpt-6-astra','llm_timeout',600,'llm_patience',3,'llm_max_calls',3);`nend`n"
        [IO.File]::WriteAllText((Join-Path $snapshot 'llm_config.m'),$safeConfig,$utf8)
        $files=@(Get-ChildItem -LiteralPath $snapshot -Recurse -File | ForEach-Object {
            @{path=$_.FullName.Substring($snapshot.Length+1);sha256=Hash $_.FullName}
        })
        Save-Json $manifestPath @{created_utc=[DateTime]::UtcNow.ToString('o');base_commit='6351644';files=$files;seeds=@(7,17,27);max_calls=3;branch_seconds=1200;min_free_gib=$MinFreeGiB}
    }
    $manifest=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach($entry in $manifest.files) {
        if((Hash (Join-Path $snapshot $entry.path)) -ne $entry.sha256) { throw 'Frozen source integrity failure.' }
    }
    if([string]::IsNullOrEmpty($HttpWorker)) { $HttpWorker=Join-Path $snapshot 'scripts/stagnation_http.ps1' }
    $env:SUC_LLM_CONFIG=Join-Path $repo 'llm_config.m'
    foreach($seed in @(7,17,27)) {
        $folder=Join-Path $Directory "seed-$seed"
        Run-Task 'capture' $seed '' (Join-Path $folder 'capture.mat')
    }
    foreach($seed in @(7,17,27)) {
        $folder=Join-Path $Directory "seed-$seed"
        $capture=Get-Content -LiteralPath (Join-Path $folder 'capture-status.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        if($capture.status -ne 'STAGNATION_CAPTURED') { continue }
        Run-Task 'prepare' $seed '' (Join-Path $folder 'request.json')
        $httpInfo=Join-Path $folder 'http-info.json'
        if(-not (Test-Path -LiteralPath $httpInfo)) {
            $ledgerPath=Join-Path $Directory 'call-ledger.json'; $claimed=$null
            if(Test-Path -LiteralPath $ledgerPath) {
                $ledger=Get-Content -LiteralPath $ledgerPath -Raw -Encoding UTF8 | ConvertFrom-Json
                $claimed=@($ledger | Where-Object directory -eq $folder)
            }
            if($claimed.Count -gt 0) {
                Save-Json $httpInfo @{status='FAILED';completed=$false;error_type='INTERRUPTED_UNKNOWN';http_seconds=600;attempt_id=$claimed[0].attempt_id}
            } else {
                Save-Json (Join-Path $Directory 'current-task.json') @{id="http-$seed";status='RUNNING';started_utc=[DateTime]::UtcNow.ToString('o')}
                try { & $HttpWorker -Directory $folder -BudgetRoot $Directory -ConfigPath $env:SUC_LLM_CONFIG }
                catch { if(-not (Test-Path -LiteralPath $httpInfo)) { throw } }
            }
        }
        Run-Task 'import' $seed '' (Join-Path $folder 'response.mat')
        $order=switch($seed) {7 {'ABCD'} 17 {'BCDA'} 27 {'CDAB'}}
        foreach($arm in $order.ToCharArray()) { Run-Task 'branch' $seed ([string]$arm) (Join-Path $folder "$arm.mat") }
    }
    foreach($seed in @(7,17,27)) {
        $folder=Join-Path $Directory "seed-$seed"
        foreach($arm in 'ABCD'.ToCharArray()) {
            $artifact=Join-Path $folder "$arm.mat"
            if(Test-Path -LiteralPath $artifact) {
                Run-Task 'diagnose' $seed ([string]$arm) (Join-Path $folder "$arm-full.mat")
            }
        }
        $capture=Get-Content -LiteralPath (Join-Path $folder 'capture-status.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        if($capture.status -eq 'STAGNATION_CAPTURED') { Run-Task 'reference' $seed '' (Join-Path $folder 'block-reference.mat') }
    }
    Run-Task 'report' 7 '' (Join-Path $Directory 'summary.json')
    $summary=Get-Content -LiteralPath (Join-Path $Directory 'summary.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if($summary.expand) {
        $phase2=Join-Path $Directory 'phase2'; [IO.Directory]::CreateDirectory($phase2) | Out-Null
        # Phase-two tasks use the same frozen source and separate six-call budget.
        $originalDirectory=$Directory; $Directory=$phase2
        foreach($name in @('tasks','logs')) { [IO.Directory]::CreateDirectory((Join-Path $Directory $name)) | Out-Null }
        foreach($seed in @(107,117,127)) {
            foreach($arm in @('A',[string]$summary.selected_mode)) {
                Run-Task 'online' $seed $arm (Join-Path $Directory "seed-$seed/$arm-online.mat")
            }
        }
        Run-Task 'online-report' 107 ([string]$summary.selected_mode) (Join-Path $Directory 'online-summary.json')
        $Directory=$originalDirectory
    }
    $index=@(Get-ChildItem -LiteralPath $Directory -Recurse -File | Where-Object {
        $_.Extension -notin @('.lock','.log') -and $_.Name -notin @('artifact-index.json','state.json','current-task.json')
    } | ForEach-Object {
        @{path=$_.FullName.Substring($Directory.Length+1);sha256=Hash $_.FullName}
    })
    Save-Json (Join-Path $Directory 'artifact-index.json') $index
    Save-Json (Join-Path $Directory 'state.json') @{status=$summary.status;ended_utc=[DateTime]::UtcNow.ToString('o');expanded=$summary.expand}
} catch {
    Save-Json (Join-Path $Directory 'state.json') @{status='STOPPED';message=$_.Exception.Message;utc=[DateTime]::UtcNow.ToString('o')}
    throw
} finally { if($null -ne $lock) { $lock.Dispose() } }
