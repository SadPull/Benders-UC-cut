param(
    [Parameter(Mandatory=$true)][string]$Directory,
    [string]$ConfigPath = (Join-Path (Split-Path $PSScriptRoot -Parent) 'llm_config.m'),
    [Parameter(Mandatory=$true)][string]$BudgetRoot,
    [ValidateRange(1,6)][int]$MaxCalls = 3,
    [ValidateRange(1,600)][int]$TimeoutSeconds = 600
)
$ErrorActionPreference = 'Stop'
$utf8 = New-Object System.Text.UTF8Encoding($false)
function Write-AtomicJson($Path, $Value) {
    $temporary = $Path + '.' + [guid]::NewGuid().ToString('N') + '.tmp'
    [System.IO.File]::WriteAllText($temporary, (ConvertTo-Json -InputObject $Value -Depth 100), $utf8)
    if ([System.IO.File]::Exists($Path)) {
        [System.IO.File]::Replace($temporary, $Path, [NullString]::Value)
    } else { [System.IO.File]::Move($temporary, $Path) }
}
function Read-ConfigString($Text, $Name) {
    $pattern = '(?m)^\s*o\.' + [regex]::Escape($Name) + '\s*=\s*''((?:''''|[^''])*)''\s*;'
    $found = [regex]::Matches($Text, $pattern)
    if ($found.Count -ne 1) { throw 'CONFIG_FORMAT' }
    return $found[0].Groups[1].Value.Replace("''", "'")
}
$lock = $null; $client = $null; $request = $null; $reply = $null
$attempt = $null; $watch = $null; $protocolWatch = $null; $entries = @(); $claimed = $false
$phase = 'INPUT'; $failure = $null
try {
    $Directory = [System.IO.Path]::GetFullPath($Directory)
    $BudgetRoot = [System.IO.Path]::GetFullPath($BudgetRoot)
    if (-not [System.IO.Directory]::Exists($Directory)) { throw 'REQUEST_DIRECTORY' }
    [System.IO.Directory]::CreateDirectory($BudgetRoot) | Out-Null
    $phase = 'LOCK'
    # Hold one global lock for the entire request, not only ledger updates.
    $lock = [System.IO.File]::Open((Join-Path $BudgetRoot 'call-ledger.lock'),
        [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
    $phase = 'INPUT'
    $infoPath = Join-Path $Directory 'http-info.json'
    $responsePath = Join-Path $Directory 'response.json'
    if ((Test-Path -LiteralPath $infoPath) -and (Test-Path -LiteralPath $responsePath)) {
        $prior = Get-Content -LiteralPath $infoPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $existingResponse = Get-Content -LiteralPath $responsePath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($prior.completed -eq $true -and $prior.status -eq 'OK' -and $existingResponse.status -eq 'completed') {
            Write-Output 'REUSED'; return
        }
    }
    $bodyText = Get-Content -LiteralPath (Join-Path $Directory 'request.json') -Raw -Encoding UTF8
    $body = $bodyText | ConvertFrom-Json
    if ($body.model -cne 'gpt-6-astra') { $phase='MODEL'; throw 'MODEL_REQUIRED' }
    $ledgerPath = Join-Path $BudgetRoot 'call-ledger.json'
    if (Test-Path -LiteralPath $ledgerPath) {
        $phase='LEDGER'
        $decoded = Get-Content -LiteralPath $ledgerPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $entries = @($decoded)
        foreach ($record in $entries) {
            if ($null -eq $record -or [string]::IsNullOrWhiteSpace([string]$record.directory) -or
                [string]::IsNullOrWhiteSpace([string]$record.attempt_id)) { throw 'INVALID_LEDGER' }
        }
    }
    if (@($entries | Where-Object { $_.directory -eq $Directory }).Count -gt 0) {
        $phase='DUPLICATE'; throw 'ALREADY_ATTEMPTED'
    }
    if ($entries.Count -ge $MaxCalls) { $phase='BUDGET'; throw 'BUDGET_EXHAUSTED' }
    $phase='CONFIG'
    $configuration = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8
    $url = Read-ConfigString $configuration 'llm_url'
    $key = Read-ConfigString $configuration 'llm_api_key'
    if ([string]::IsNullOrWhiteSpace($key)) { throw 'EMPTY_KEY' }
    $uri = [uri]$url
    if (-not $uri.IsAbsoluteUri -or $uri.Scheme -notin @('https','http')) { throw 'URL_FORMAT' }
    Add-Type -AssemblyName System.Net.Http
    $client = New-Object System.Net.Http.HttpClient
    $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSeconds)
    $request = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Post, $uri)
    $request.Headers.Authorization = New-Object System.Net.Http.Headers.AuthenticationHeaderValue('Bearer', $key)
    $request.Content = New-Object System.Net.Http.StringContent($bodyText, $utf8, 'application/json')
    $configuration=$null; $key=$null; $url=$null
    $attempt = [guid]::NewGuid().ToString('N')
    $entry = [pscustomobject]@{attempt_id=$attempt; directory=$Directory;
        started_utc=[DateTime]::UtcNow.ToString('o'); status='CLAIMED'}
    $entries += $entry
    $phase='LEDGER'; Write-AtomicJson $ledgerPath $entries; $claimed=$true
    $phase='HTTP'; $watch=[Diagnostics.Stopwatch]::StartNew()
    $reply = $client.SendAsync($request).GetAwaiter().GetResult()
    $responseText = $reply.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    $watch.Stop()
    if (-not $reply.IsSuccessStatusCode) { $failure='HTTP_STATUS'; throw 'HTTP_STATUS' }
    $phase='RESPONSE'; $protocolWatch=[Diagnostics.Stopwatch]::StartNew()
    $response = $responseText | ConvertFrom-Json
    $protocolWatch.Stop()
    if ($response.status -ne 'completed') { $failure='INCOMPLETE_RESPONSE'; throw 'INCOMPLETE_RESPONSE' }
    # Only successful protocol responses are stored; error bodies are discarded.
    Write-AtomicJson $responsePath $response
    $info=[ordered]@{status='OK'; completed=$true; error_type=''; http_seconds=$watch.Elapsed.TotalSeconds;
        protocol_seconds=$protocolWatch.Elapsed.TotalSeconds; attempt_id=$attempt}
    Write-AtomicJson $infoPath $info
    $entry.status='OK'; Write-AtomicJson $ledgerPath $entries
    Write-Output 'OK'
} catch {
    if ($null -ne $watch -and $watch.IsRunning) { $watch.Stop() }
    if ($null -eq $failure) {
        $failure = $phase + '_FAILED'
        if ($phase -eq 'HTTP' -and $watch.Elapsed.TotalSeconds -ge $TimeoutSeconds) { $failure='TIMEOUT' }
    }
    if ($claimed) {
        $elapsed=0; if ($null -ne $watch) { $elapsed=$watch.Elapsed.TotalSeconds }
        $protocolElapsed=0
        if ($null -ne $protocolWatch) { $protocolWatch.Stop(); $protocolElapsed=$protocolWatch.Elapsed.TotalSeconds }
        $info=[ordered]@{status='FAILED'; completed=$false; error_type=$failure; http_seconds=$elapsed;
            protocol_seconds=$protocolElapsed; attempt_id=$attempt}
        try { Write-AtomicJson $infoPath $info; $entry.status='FAILED'; Write-AtomicJson $ledgerPath $entries } catch { }
    }
    # Never forward exception text: HTTP exceptions can contain URLs or headers.
    throw ('stagnation_http: ' + $failure)
} finally {
    if ($null -ne $reply) { $reply.Dispose() }
    if ($null -ne $request) { $request.Dispose() }
    if ($null -ne $client) { $client.Dispose() }
    if ($null -ne $lock) { $lock.Dispose() }
}
