param(
    [Parameter(Mandatory=$true)][string]$Directory,
    [switch]$WriteIndex
)
# Default operation only reads artifacts and process metadata; no model/config access.
$ErrorActionPreference='Stop'
$Directory=[IO.Path]::GetFullPath($Directory).TrimEnd('\','/')
$issues=New-Object 'System.Collections.Generic.List[string]'
$warnings=New-Object 'System.Collections.Generic.List[string]'
$active=@(); $doneCount=0; $manifestCount=0; $budgets=@(); $indexLock=$null
function Read-Json($Path) { Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json }
function Hash($Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash }
function Inside($Path) {
    [IO.Path]::GetFullPath($Path).StartsWith($Directory+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)
}
function Check-Hash($Path,$Expected,$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { $issues.Add('Missing: '+$Label); return }
    if ((Hash $Path) -ne $Expected) { $issues.Add('Hash mismatch: '+$Label) }
}
function Audit-Budget($Root,$Limit,$Online) {
    $ledgerPath=Join-Path $Root 'call-ledger.json'; $entries=@()
    if (Test-Path -LiteralPath $ledgerPath) { $decoded=Read-Json $ledgerPath; $entries=@($decoded) }
    if ($entries.Count -gt $Limit) { $issues.Add('Call limit exceeded: '+$Root) }
    $ids=@{}; $dirs=@{}
    foreach ($entry in $entries) {
        if (-not $entry.attempt_id -or -not $entry.directory) { $issues.Add('Malformed ledger entry: '+$Root); continue }
        if ($ids.ContainsKey([string]$entry.attempt_id) -or $dirs.ContainsKey([string]$entry.directory)) { $issues.Add('Duplicate ledger claim: '+$Root) }
        $ids[[string]$entry.attempt_id]=$true; $dirs[[string]$entry.directory]=$true
        if (-not (Inside $entry.directory)) { $issues.Add('Ledger directory outside experiment'); continue }
        $infoPath=Join-Path $entry.directory 'http-info.json'
        if ($Online) { $infoPath=Join-Path $entry.directory 'info.json' }
        if (-not (Test-Path -LiteralPath $infoPath)) {
            if ($entry.status -eq 'CLAIMED') { $warnings.Add('Claim without response yet: '+$entry.attempt_id) }
            else { $issues.Add('Missing call info: '+$entry.attempt_id) }
            continue
        }
        $info=Read-Json $infoPath
        if (-not $Online -and $info.attempt_id -ne $entry.attempt_id) { $issues.Add('Call identity mismatch: '+$entry.attempt_id) }
        if ($info.status -ne $entry.status) { $issues.Add('Call status mismatch: '+$entry.attempt_id) }
        $requestPath=Join-Path $entry.directory 'request.json'
        if (-not (Test-Path -LiteralPath $requestPath)) { $issues.Add('Missing request: '+$entry.attempt_id) }
        else {
            $request=Read-Json $requestPath
            if ($request.model -cne 'gpt-6-astra') { $issues.Add('Unexpected model: '+$entry.attempt_id) }
        }
        if ($info.status -eq 'OK') {
            $responsePath=Join-Path $entry.directory 'response.json'
            if (-not (Test-Path -LiteralPath $responsePath)) { $issues.Add('Missing response: '+$entry.attempt_id) }
            else {
                $response=Read-Json $responsePath
                if ($response.status -ne 'completed' -or -not ($response.PSObject.Properties.Name -contains 'output')) {
                    $issues.Add('Incomplete response protocol: '+$entry.attempt_id)
                }
            }
            if (-not $Online -and -not $info.completed) { $issues.Add('Success without completed flag: '+$entry.attempt_id) }
        }
    }
    [pscustomobject]@{root=$Root;used=$entries.Count;limit=$Limit;statuses=@($entries | ForEach-Object {$_.status})}
}
try {
    $manifest=Read-Json (Join-Path $Directory 'manifest.json')
    foreach ($entry in $manifest.files) {
        $path=Join-Path (Join-Path $Directory 'snapshot') $entry.path
        if (-not (Inside $path)) { $issues.Add('Manifest path outside experiment'); continue }
        Check-Hash $path $entry.sha256 ('snapshot/'+$entry.path); $manifestCount++
    }
    if ($manifest.max_calls -ne 3 -or (@($manifest.seeds) -join ',') -ne '7,17,27') { $issues.Add('Initial experiment budget or seed manifest differs') }
    foreach ($root in @($Directory,(Join-Path $Directory 'phase2'))) {
        $currentPath=Join-Path $root 'current-task.json'
        if (Test-Path -LiteralPath $currentPath) {
            $current=Read-Json $currentPath
            if ($current.pid) {
                $process=Get-Process -Id $current.pid -ErrorAction SilentlyContinue
                if ($process -and $process.StartTime.ToUniversalTime().Ticks -eq ([DateTime]$current.process_started_utc).ToUniversalTime().Ticks) {
                    $active += [pscustomobject]@{task=$current.id;pid=$current.pid;started_utc=$current.process_started_utc}
                }
            }
        }
        $tasks=Join-Path $root 'tasks'; if (-not (Test-Path -LiteralPath $tasks)) { continue }
        foreach ($file in Get-ChildItem -LiteralPath $tasks -Filter '*.done.json') {
            $done=Read-Json $file.FullName
            if ($done.id -notmatch '^(capture|prepare|import|branch|diagnose|reference|report|online|online-report)-(\d+)-(.*)$') {
                $issues.Add('Unknown completion task: '+$file.Name); continue
            }
            $action=$Matches[1]; $seed=$Matches[2]; $mode=$Matches[3]
            $folder=Join-Path $root ('seed-'+$seed)
            $artifact=switch ($action) {
                'capture' { Join-Path $folder 'capture.mat' }
                'prepare' { Join-Path $folder 'request.json' }
                'import' { Join-Path $folder 'response.mat' }
                'branch' { Join-Path $folder ($mode+'.mat') }
                'diagnose' { Join-Path $folder ($mode+'-full.mat') }
                'reference' { Join-Path $folder 'block-reference.mat' }
                'report' { Join-Path $root 'summary.json' }
                'online' { Join-Path $folder ($mode+'-online.mat') }
                'online-report' { Join-Path $root 'online-summary.json' }
            }
            if ($done.status -ne 'COMPLETE') { $issues.Add('Invalid done status: '+$done.id) }
            Check-Hash $artifact $done.sha256 $done.id; $doneCount++
            if ($action -eq 'capture' -and -not (Test-Path -LiteralPath (Join-Path $folder 'capture-status.json'))) { $issues.Add('Missing capture status: '+$done.id) }
            if ($action -eq 'prepare' -and -not (Test-Path -LiteralPath (Join-Path $folder 'request-info.mat'))) { $issues.Add('Missing request timing: '+$done.id) }
        }
    }
    if ($active.Count -gt 1) { $issues.Add('Multiple controlled numerical tasks active') }
    $budgets += Audit-Budget $Directory 3 $false
    $onlineRoot=Join-Path $Directory 'phase2/online-calls'
    if (Test-Path -LiteralPath $onlineRoot) { $budgets += Audit-Budget $onlineRoot 6 $true }
    if ($WriteIndex) {
        if ($issues.Count -gt 0 -or $active.Count -gt 0) { throw 'Index requires successful audit and no active controlled task.' }
        # The controller owns this lock until all its final writes are complete.
        $indexLock=[IO.File]::Open((Join-Path $Directory 'runner.lock'),'OpenOrCreate','ReadWrite','None')
        $state=Read-Json (Join-Path $Directory 'state.json')
        if ($state.status -notin @('COMPLETE','FAILED')) { throw 'Index requires a completed experiment state.' }
        $indexPath=Join-Path $Directory 'artifact-index.json'
        $files=@(Get-ChildItem -LiteralPath $Directory -Recurse -File | Where-Object {
            $_.Extension -notin @('.lock','.log') -and
            $_.Name -notin @('artifact-index.json','artifact-index.json.tmp','state.json','current-task.json')
        } | ForEach-Object { [pscustomobject]@{path=$_.FullName.Substring($Directory.Length+1);sha256=Hash $_.FullName} })
        $utf8=New-Object Text.UTF8Encoding($false)
        [IO.File]::WriteAllText($indexPath+'.tmp',(ConvertTo-Json -InputObject $files -Depth 10),$utf8)
        if (Test-Path -LiteralPath $indexPath) { [IO.File]::Replace($indexPath+'.tmp',$indexPath,[NullString]::Value) }
        else { [IO.File]::Move($indexPath+'.tmp',$indexPath) }
    }
    [pscustomobject]@{ok=($issues.Count -eq 0);manifest_files_checked=$manifestCount;done_artifacts_checked=$doneCount;
        budgets=$budgets;active_controlled=$active;issues=@($issues.ToArray());warnings=@($warnings.ToArray());index_written=[bool]$WriteIndex} |
        ConvertTo-Json -Depth 10
} finally { if ($null -ne $indexLock) { $indexLock.Dispose() } }
