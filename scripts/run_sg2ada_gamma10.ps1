# Second official StyleGAN2-ADA baseline run: R1 gamma 10, 3,000 kimg, offline,
# with an automatic stop rule (article/gpu_prompt_baseline_gamma10.txt).
#
# Identical to scripts/run_sg2ada_baseline.bat (first run) except
#     --gamma=10  --kimg=3000  --desc=baseline-gamma10
# Same clone, venv, dataset zip, outdir, cache dir, dead proxy and PATH prefix.
#
# train.py runs as a child process (cmd.exe /c python ... > log 2>&1). Every 60 s
# this script re-reads metric-fid50k_full.jsonl of the new run folder. Right after
# a new line appears (the snapshot pickle for that line is already on disk, because
# training_loop.py saves the pickle before it evaluates the metrics) it applies:
#   rule b  latest FID > 3 x lowest FID so far                       -> collapse
#   rule a  lowest FID so far has not improved by more than 2% over
#           the last three snapshots, i.e. best_now > 0.98 * best three
#           snapshots ago (600 kimg)                                  -> plateau
# and kills the training process tree if either fires. Otherwise training ends
# by itself at 3,000 kimg (or on an error exit).
#
# Usage:
#   pwsh -File scripts\run_sg2ada_gamma10.ps1              real run
#   pwsh -File scripts\run_sg2ada_gamma10.ps1 -TestRule    only test the stop rule on the first run's FIDs
#   pwsh -File scripts\run_sg2ada_gamma10.ps1 -SmokeTest   stop by itself after the first FID line (kimg 0)
# Works with Windows PowerShell 5.1 as well.

param(
    [switch]$TestRule,
    [switch]$SmokeTest
)

$ErrorActionPreference = 'Stop'
# print numbers with a decimal point regardless of the Windows locale
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentUICulture = [System.Globalization.CultureInfo]::InvariantCulture

# ---------------------------------------------------------------- settings
$SG3        = (Resolve-Path (Join-Path $PSScriptRoot '..\..\stylegan3')).Path   # NVlabs stylegan3 clone next to this repository
$OUTDIR_REL = '../sg2ada_runs'                    # relative to $SG3, exactly as in the first run
$OUTDIR     = Join-Path (Split-Path $SG3 -Parent) 'sg2ada_runs'
$DATA       = '../sg2ada_data/catalogue_512.zip'
$DESC       = 'baseline-gamma10'
$GAMMA      = 10
$TOTAL_KIMG = 3000
$POLL_SEC   = 60
$RULE_A_WINDOW = 3        # snapshots
$RULE_A_MIN_IMPROVEMENT = 0.02
$RULE_B_FACTOR = 3.0

$TRAIN_ARGS = "--outdir=$OUTDIR_REL --data=$DATA --cfg=stylegan2 --gpus=1 --batch=16 --mirror=1 --aug=ada --gamma=$GAMMA --metrics=fid50k_full --kimg=$TOTAL_KIMG --snap=50 --desc=$DESC"

# ---------------------------------------------------------------- stop rule
# Input: FID values in snapshot order. Output: the decision for the LAST value,
# evaluated as if every earlier value had already been seen (the function is pure,
# so it is called on the full list after every new line).
function Get-StopDecision {
    param([double[]]$Fids)
    $n = $Fids.Count
    $result = [ordered]@{ Reason = ''; Index = -1; Best = $null; BestIndex = -1 }
    if ($n -eq 0) { return $result }
    $runningBest = New-Object double[] $n
    $best = [double]::PositiveInfinity
    $bestIndex = -1
    for ($i = 0; $i -lt $n; $i++) {
        $f = $Fids[$i]
        if ($f -lt $best) { $best = $f; $bestIndex = $i }
        $runningBest[$i] = $best
        $result.Best = $best
        $result.BestIndex = $bestIndex
        # rule b: collapse
        if ($f -gt $RULE_B_FACTOR * $best) {
            $result.Reason = 'rule b'
            $result.Index = $i
            return $result
        }
        # rule a: plateau over the last $RULE_A_WINDOW snapshots
        if ($i -ge $RULE_A_WINDOW) {
            $ref = $runningBest[$i - $RULE_A_WINDOW]
            if ($best -gt (1.0 - $RULE_A_MIN_IMPROVEMENT) * $ref) {
                $result.Reason = 'rule a'
                $result.Index = $i
                return $result
            }
        }
    }
    return $result
}

function Test-StopRule {
    $fids = @(305.8, 59.4, 32.8, 28.9, 25.6, 21.3, 17.0, 9.28, 9.06, 219.1)
    Write-Host "Stop-rule test on the first run's FID values (snapshots 0..$(($fids.Count-1)*200) kimg):"
    $ok = $true
    for ($k = 1; $k -le $fids.Count; $k++) {
        $d = Get-StopDecision -Fids $fids[0..($k-1)]
        $line = "  after {0,2} lines (kimg {1,5}): fid {2,7:N2}  best {3,7:N2} (kimg {4,5})  decision: {5}" -f $k, (($k-1)*200), $fids[$k-1], $d.Best, ($d.BestIndex*200), $(if ($d.Reason) { $d.Reason } else { 'continue' })
        Write-Host $line
        if ($k -lt $fids.Count -and $d.Reason) { $ok = $false }
    }
    $final = Get-StopDecision -Fids $fids
    $pass = ($final.Reason -eq 'rule b') -and ($final.Index -eq 9) -and ([math]::Abs($final.Best - 9.06) -lt 1e-9) -and ($final.BestIndex -eq 8) -and $ok
    # extra cases
    $plateau = @(100, 50, 30, 20, 19.8, 19.7, 19.65)       # best 20 -> 19.65 is 1.75% over 3 snapshots -> rule a
    $dp = Get-StopDecision -Fids $plateau
    $slow = @(100, 50, 30, 20, 19.7, 19.4, 19.1)           # 20 -> 19.1 is 4.5% over 3 snapshots -> continue
    $ds = Get-StopDecision -Fids $slow
    Write-Host ("  extra: plateau case {0} -> {1} (expect rule a); slow-but-improving case {2} -> {3} (expect continue)" -f ($plateau -join ','), $dp.Reason, ($slow -join ','), $(if ($ds.Reason) { $ds.Reason } else { 'continue' }))
    $pass = $pass -and ($dp.Reason -eq 'rule a') -and ($dp.Index -eq 6) -and ($ds.Reason -eq '')
    Write-Host ("Stop-rule test: " + $(if ($pass) { 'PASS' } else { 'FAIL' }) + " (expected: rule b at the 219.1 line, best 9.06 at kimg 1600)")
    return $pass
}

if ($TestRule) {
    $pass = Test-StopRule
    if ($pass) { exit 0 } else { exit 1 }
}

# ---------------------------------------------------------------- helpers
$TS = Get-Date -Format yyyyMMdd_HHmmss
$TRAIN_LOG   = Join-Path $OUTDIR "train_gamma10_$TS.log"
$MONITOR_LOG = Join-Path $OUTDIR "monitor_gamma10_$TS.log"

function Log {
    param([string]$Msg)
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Msg
    Write-Host $line
    Add-Content -Path $MONITOR_LOG -Value $line
}

function Read-Fids {
    param([string]$JsonlPath)
    $out = @()
    if (-not (Test-Path $JsonlPath)) { return $out }
    foreach ($raw in (Get-Content -Path $JsonlPath -ErrorAction SilentlyContinue)) {
        if (-not $raw.Trim()) { continue }
        try {
            $o = $raw | ConvertFrom-Json
            $out += [pscustomobject]@{ Snapshot = $o.snapshot_pkl; Fid = [double]$o.results.fid50k_full }
        } catch {
            # a line still being written; it will be complete at the next poll
        }
    }
    return $out
}

function Stop-Tree {
    param([System.Diagnostics.Process]$Proc)
    if ($Proc -and -not $Proc.HasExited) {
        Log "Killing training process tree (cmd pid $($Proc.Id))"
        & taskkill /PID $Proc.Id /T /F 2>&1 | ForEach-Object { Log "  taskkill: $_" }
        try { $Proc.WaitForExit(60000) | Out-Null } catch {}
    }
}

# ---------------------------------------------------------------- environment (as in the first run)
$env:DNNLIB_CACHE_DIR = Join-Path $env:USERPROFILE '.cache\dnnlib'
$env:HTTP_PROXY  = 'http://127.0.0.1:9'
$env:HTTPS_PROXY = 'http://127.0.0.1:9'
$env:NO_PROXY    = ''
$env:PATH = "$SG3\.venv\Scripts;" + $env:PATH
if (-not (Test-Path $OUTDIR)) { New-Item -ItemType Directory -Path $OUTDIR | Out-Null }

# ---------------------------------------------------------------- start training
$startTime = Get-Date
$python = "$SG3\.venv\Scripts\python.exe"
$cmdLine = "/c `"`"$python`" -u train.py $TRAIN_ARGS > `"$TRAIN_LOG`" 2>&1`""
Log "Second baseline run (gamma $GAMMA, $TOTAL_KIMG kimg). Script: $($MyInvocation.MyCommand.Path)"
Log "Train log:   $TRAIN_LOG"
Log "Monitor log: $MONITOR_LOG"
Log "Command (cwd $SG3): python.exe -u train.py $TRAIN_ARGS"
if ($SmokeTest) { Log "SMOKE TEST: will stop after the first fid50k_full line" }

$proc = Start-Process -FilePath 'cmd.exe' -ArgumentList $cmdLine -WorkingDirectory $SG3 -PassThru -NoNewWindow
Log "Started cmd pid $($proc.Id)"

$runDir = $null
$jsonl = $null
$seen = 0
$decision = $null
$reason = ''
$fids = @()

try {
    while ($true) {
        # locate the run folder once it exists
        if (-not $runDir) {
            $cands = Get-ChildItem -Path $OUTDIR -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match "^\d{5}-.*-$([regex]::Escape($DESC))$" -and $_.CreationTime -ge $startTime.AddMinutes(-1) } |
                Sort-Object CreationTime
            if ($cands) {
                $runDir = $cands[-1].FullName
                $jsonl = Join-Path $runDir 'metric-fid50k_full.jsonl'
                Log "Run folder: $runDir"
            }
        }

        if ($jsonl) {
            $rows = @(Read-Fids -JsonlPath $jsonl)
            if ($rows.Count -gt $seen) {
                for ($i = $seen; $i -lt $rows.Count; $i++) {
                    Log ("New FID line {0}: {1}  fid50k_full = {2:N3}" -f ($i+1), $rows[$i].Snapshot, $rows[$i].Fid)
                }
                $seen = $rows.Count
                $fids = @($rows | ForEach-Object { $_.Fid })
                $d = Get-StopDecision -Fids $fids
                Log ("  lowest so far {0:N3} ({1})  decision: {2}" -f $d.Best, $rows[$d.BestIndex].Snapshot, $(if ($d.Reason) { $d.Reason } else { 'continue' }))
                if ($d.Reason) {
                    $decision = $d
                    $reason = $d.Reason
                    Stop-Tree -Proc $proc
                    break
                }
                if ($SmokeTest) {
                    $decision = $d
                    $reason = 'smoke test'
                    Stop-Tree -Proc $proc
                    break
                }
            }
        }

        if ($proc.HasExited) {
            # one last read in case the final line arrived just before exit
            if ($jsonl) {
                $rows = @(Read-Fids -JsonlPath $jsonl)
                for ($i = $seen; $i -lt $rows.Count; $i++) {
                    Log ("New FID line {0}: {1}  fid50k_full = {2:N3}" -f ($i+1), $rows[$i].Snapshot, $rows[$i].Fid)
                }
                $seen = $rows.Count
                $fids = @($rows | ForEach-Object { $_.Fid })
            }
            $decision = Get-StopDecision -Fids $fids
            if ($decision.Reason) { $reason = $decision.Reason }
            elseif ($proc.ExitCode -eq 0) { $reason = 'kimg limit' }
            else { $reason = "train.py exited with code $($proc.ExitCode)" }
            Log "Training process exited with code $($proc.ExitCode)"
            break
        }

        Start-Sleep -Seconds $POLL_SEC
    }
}
finally {
    Stop-Tree -Proc $proc
}

# ---------------------------------------------------------------- summary
$elapsed = (Get-Date) - $startTime
$lines = @()
$lines += "STOP SUMMARY  ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))"
$lines += "Run folder:   $runDir"
$lines += "Stop reason:  $reason"
$lines += ("Elapsed:      {0:dd\.hh\:mm\:ss}" -f $elapsed)
$lines += "Snapshots and fid50k_full:"
$rows = @()
if ($jsonl) { $rows = @(Read-Fids -JsonlPath $jsonl) }
for ($i = 0; $i -lt $rows.Count; $i++) {
    $mark = ''
    if ($decision -and $i -eq $decision.BestIndex) { $mark = '  <-- lowest' }
    if ($decision -and $decision.Reason -and $i -eq $decision.Index) { $mark += "  <-- $($decision.Reason) triggered here" }
    $lines += ("  {0,-30} {1,10:N3}{2}" -f $rows[$i].Snapshot, $rows[$i].Fid, $mark)
}
if ($decision -and $decision.BestIndex -ge 0 -and $rows.Count -gt 0) {
    $lines += ("Lowest FID:   {0:N3}  ->  {1}" -f $decision.Best, (Join-Path $runDir $rows[$decision.BestIndex].Snapshot))
} else {
    $lines += "Lowest FID:   (no FID line written)"
}
foreach ($l in $lines) { Log $l }
if ($runDir -and (Test-Path $runDir)) {
    $summaryPath = Join-Path $runDir 'stop_summary.txt'
    $lines | Set-Content -Path $summaryPath
    Log "Summary written to $summaryPath"
}
