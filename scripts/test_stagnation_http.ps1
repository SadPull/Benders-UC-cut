# Offline guards only: every case exits before HttpClient construction.
$ErrorActionPreference='Stop'
$worker=Join-Path $PSScriptRoot 'stagnation_http.ps1'
$testRoot=Join-Path ([System.IO.Path]::GetTempPath()) ('stagnation-http-test-' + [guid]::NewGuid().ToString('N'))
[System.IO.Directory]::CreateDirectory($testRoot) | Out-Null
function Assert-True($Condition,$Message) { if (-not $Condition) { throw $Message } }
function Expect-Failure($Directory,$Budget,$Expected) {
    $caught=$false
    try { & $worker -Directory $Directory -BudgetRoot $Budget -ConfigPath (Join-Path $testRoot 'nonexistent-config.m') }
    catch { $caught=$true; Assert-True ($_.Exception.Message -match $Expected) ('Wrong error: ' + $_.Exception.Message) }
    Assert-True $caught 'Expected failure did not occur.'
}
try {
    # Load only the real helper's AST, never the worker's request-execution body.
    $tokens=$null; $parseErrors=$null
    $ast=[System.Management.Automation.Language.Parser]::ParseFile($worker,[ref]$tokens,[ref]$parseErrors)
    Assert-True ($parseErrors.Count -eq 0) 'Worker has PowerShell syntax errors.'
    $definition=$ast.Find({param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Write-AtomicJson'
    },$true)
    Assert-True ($null -ne $definition) 'Missing atomic JSON helper.'
    . ([scriptblock]::Create($definition.Extent.Text))
    $utf8=New-Object System.Text.UTF8Encoding($false)
    $atomicPath=Join-Path $testRoot 'atomic.json'
    $unicode=[string][char]0x4E2D
    Write-AtomicJson $atomicPath @{status='CLAIMED';message=$unicode}
    Write-AtomicJson $atomicPath @{status='OK';message=$unicode}
    $roundtrip=Get-Content -LiteralPath $atomicPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert-True ($roundtrip.status -eq 'OK' -and $roundtrip.message -ceq $unicode) 'Atomic overwrite or UTF8 roundtrip failed.'
    Write-AtomicJson $atomicPath @(@{attempt_id=1},@{attempt_id=2})
    $roundtrip=Get-Content -LiteralPath $atomicPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert-True ($roundtrip.Count -eq 2) 'Second ledger overwrite failed.'
    Assert-True (@(Get-ChildItem -LiteralPath $testRoot -Filter 'atomic.json.*.tmp').Count -eq 0) 'Atomic replacement left temporary files.'
    $directory=Join-Path $testRoot 'request'; $budget=Join-Path $testRoot 'budget'
    [System.IO.Directory]::CreateDirectory($directory) | Out-Null
    Set-Content -LiteralPath (Join-Path $directory 'request.json') -Value '{"model":"wrong-model"}'
    Expect-Failure $directory $budget 'MODEL_FAILED'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $budget 'call-ledger.json'))) 'Wrong model consumed budget.'
    Set-Content -LiteralPath (Join-Path $directory 'request.json') -Value '{"model":"gpt-6-astra"}'
    $entry=[pscustomobject]@{directory=$directory; attempt_id='offline'; status='CLAIMED'}
    ConvertTo-Json -InputObject @($entry) | Set-Content -LiteralPath (Join-Path $budget 'call-ledger.json')
    Expect-Failure $directory $budget 'DUPLICATE_FAILED'
    $entries=1..3 | ForEach-Object { [pscustomobject]@{directory=('other-' + $_); attempt_id=$_; status='FAILED'} }
    ConvertTo-Json -InputObject @($entries) | Set-Content -LiteralPath (Join-Path $budget 'call-ledger.json')
    Expect-Failure $directory $budget 'BUDGET_FAILED'
    $lock=[System.IO.File]::Open((Join-Path $budget 'call-ledger.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    try { Expect-Failure $directory $budget 'LOCK_FAILED' } finally { $lock.Dispose() }
    Set-Content -LiteralPath (Join-Path $directory 'response.json') -Value '{"status":"completed","output":[]}'
    Set-Content -LiteralPath (Join-Path $directory 'http-info.json') -Value '{"status":"OK","completed":true,"http_seconds":1,"attempt_id":"offline"}'
    $result=& $worker -Directory $directory -BudgetRoot $budget -ConfigPath (Join-Path $testRoot 'nonexistent-config.m')
    Assert-True ($result -eq 'REUSED') 'Complete response was not reused.'
    Write-Output 'PASS: real helper atomic overwrites and UTF8, model guard, duplicate claim, exhausted budget, lock, response reuse; no HTTP or real config access.'
} finally {
    $resolved=[System.IO.Path]::GetFullPath($testRoot)
    $temp=[System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
    if ($resolved.StartsWith($temp,[System.StringComparison]::OrdinalIgnoreCase) -and
        ([System.IO.Path]::GetFileName($resolved)).StartsWith('stagnation-http-test-')) {
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
}
