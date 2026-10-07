# Run under Windows PowerShell 5.1 and PowerShell 7; no controller, MATLAB or HTTP execution.
$ErrorActionPreference='Stop'
function Assert-True($Condition,$Message) { if (-not $Condition) { throw $Message } }
function Read-Comparison($Path,$Pattern) {
    $tokens=$null; $parseErrors=$null
    $ast=[System.Management.Automation.Language.Parser]::ParseFile($Path,[ref]$tokens,[ref]$parseErrors)
    Assert-True ($parseErrors.Count -eq 0) 'Script has PowerShell syntax errors.'
    $nodes=@($ast.FindAll({param($node)
        $node -is [System.Management.Automation.Language.BinaryExpressionAst] -and
        $node.Operator -in @('Ieq','Ige') -and $node.Extent.Text -match $Pattern
    },$true))
    Assert-True ($nodes.Count -eq 1) ('Expected one timestamp comparison: ' + $Pattern)
    return [scriptblock]::Create($nodes[0].Extent.Text)
}
$auditCheck=Read-Comparison (Join-Path $PSScriptRoot 'audit_stagnation.ps1') '\$current\.process_started_utc'
$runnerCheck=Read-Comparison (Join-Path $PSScriptRoot 'run_stagnation.ps1') '\$current\.process_started_utc'
$recoveryCheck=Read-Comparison (Join-Path $PSScriptRoot 'run_stagnation.ps1') '\$startedRecord\.started_utc'
$stamp='2026-10-07T05:20:48.0627211Z'
$expected=[DateTimeOffset]::Parse($stamp).UtcDateTime
$jsonRecord=('{"pid":123,"process_started_utc":"'+$stamp+'","started_utc":"'+$stamp+'"}') | ConvertFrom-Json
$records=@($jsonRecord,
    [pscustomobject]@{pid=123;process_started_utc=$stamp;started_utc=$stamp},
    [pscustomobject]@{pid=123;process_started_utc=$expected;started_utc=$expected})
$success=[IO.Path]::GetTempFileName()
try {
    foreach($record in $records) {
        $current=$record; $startedRecord=$record
        $process=[pscustomobject]@{Id=123;StartTime=$expected}; $active=$process
        Assert-True (& $auditCheck) 'Audit rejected the matching process start.'
        Assert-True (& $runnerCheck) 'Controller rejected the matching process start.'
        $process.StartTime=$expected.ToLocalTime()
        Assert-True (& $auditCheck) 'Audit rejected the same instant in local time.'
        Assert-True (& $runnerCheck) 'Controller rejected the same instant in local time.'
        $process.StartTime=$expected.AddTicks(1)
        Assert-True (-not (& $auditCheck)) 'Audit accepted a reused PID with a different start.'
        Assert-True (-not (& $runnerCheck)) 'Controller accepted a reused PID with a different start.'
        [IO.File]::SetLastWriteTimeUtc($success,$expected.AddSeconds(-1))
        Assert-True (-not (& $recoveryCheck)) 'Recovery accepted a success marker older than the task.'
        [IO.File]::SetLastWriteTimeUtc($success,$expected)
        Assert-True (& $recoveryCheck) 'Recovery rejected a success marker at the task start.'
        [IO.File]::SetLastWriteTimeUtc($success,$expected.AddSeconds(1))
        Assert-True (& $recoveryCheck) 'Recovery rejected a success marker after the task start.'
    }
    Write-Output ('PASS: 27 checks of real timestamp expressions; PowerShell '+$PSVersionTable.PSVersion+
        '; JSON timestamp type '+$jsonRecord.started_utc.GetType().FullName+'; no MATLAB, HTTP or config access.')
} finally { Remove-Item -LiteralPath $success -Force }
