#!/usr/bin/env pwsh
# watch_pipeline.ps1 — live monitor for upscale.py pipeline
# Polls upscale_log.txt, shows stage transitions, alerts on crash or completion.
# Run in a SEPARATE terminal: .\watch_pipeline.ps1

param(
    [string]$Log = "C:\VideoUpscale\upscale_log.txt",
    [int]$PollSec = 30
)

$lastStage  = ""
$startTime  = Get-Date
$lastLine   = 0

# Skip lines already in the log (old runs / prior crashes)
if (Test-Path $Log) {
    $existing = Get-Content $Log -ErrorAction SilentlyContinue
    if ($existing) { $lastLine = $existing.Count }
}

function Beep([int]$freq=880, [int]$ms=400) { [Console]::Beep($freq, $ms) }

Write-Host "=== VideoUpscale Pipeline Watcher ===" -ForegroundColor Cyan
Write-Host "Log : $Log"
Write-Host "Poll: every ${PollSec}s   Started: $($startTime.ToString('HH:mm:ss'))   Skipping $lastLine existing lines"
Write-Host ("-" * 60)

while ($true) {
    Start-Sleep -Seconds $PollSec

    if (-not (Test-Path $Log)) {
        Write-Host "[$(Get-Date -F 'HH:mm:ss')] Waiting for log file..." -ForegroundColor DarkGray
        continue
    }

    $lines = Get-Content $Log -ErrorAction SilentlyContinue
    if (-not $lines) { continue }

    # Only process lines added since last poll
    if ($lines.Count -le $lastLine) { continue }
    $newLines = @($lines[$lastLine..($lines.Count - 1)])
    $lastLine  = $lines.Count

    $elapsed = [math]::Round(((Get-Date) - $startTime).TotalMinutes, 1)

    # --- Stage transitions (exact prefix match, never matches "Done in ...") ---
    foreach ($l in $newLines) {
        if ($l -match '^\s+Stage\s+\d') {
            $stage = $l.Trim()
            if ($stage -ne $lastStage) {
                $lastStage = $stage
                Write-Host ""
                Write-Host "[$(Get-Date -F 'HH:mm:ss')] >>> $stage" -ForegroundColor Green
            }
        }
    }

    # --- Pipeline complete: "  DONE!  output/..." ---
    $doneLine = $newLines | Where-Object { $_ -match 'DONE!' } | Select-Object -First 1
    if ($doneLine) {
        Write-Host ""
        Write-Host "[$(Get-Date -F 'HH:mm:ss')] ### PIPELINE COMPLETE ###" -ForegroundColor Green
        Write-Host "  $($doneLine.Trim())" -ForegroundColor Green
        Write-Host "Total elapsed: ${elapsed} min" -ForegroundColor Cyan
        Beep 880 300; Start-Sleep -Milliseconds 100; Beep 1100 300; Start-Sleep -Milliseconds 100; Beep 1320 500
        break
    }

    # --- Crash detection (Python traceback lines) ---
    $crashLine = $newLines | Where-Object { $_ -match 'Traceback \(most recent call last\)|subprocess\.CalledProcessError' } | Select-Object -First 1
    if ($crashLine) {
        Write-Host ""
        Write-Host "[$(Get-Date -F 'HH:mm:ss')] !!! CRASH DETECTED !!!" -ForegroundColor Red
        Write-Host "Last 25 log lines:" -ForegroundColor Yellow
        $lines | Select-Object -Last 25 | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
        Beep 440 600; Start-Sleep -Milliseconds 200; Beep 440 600; Start-Sleep -Milliseconds 200; Beep 440 600
        Write-Host ""
        Write-Host "Restart command:" -ForegroundColor Red
        Write-Host "  C:\VideoUpscale\venv\Scripts\python.exe upscale.py `"input\14238437_1080_1920_30fps.mp4`" --model Real_HAT_GAN_SRx4_sharper --scale 4 *>> upscale_log.txt" -ForegroundColor Cyan
        break
    }

    # --- Live progress ---
    $progLine = $newLines | Where-Object { $_ -match 'frame=|it/s\]|inference' } | Select-Object -Last 1
    if (-not $progLine) { $progLine = $lines | Select-Object -Last 1 }
    Write-Host "[$(Get-Date -F 'HH:mm:ss')] +${elapsed}m  $($progLine.Trim())" -ForegroundColor DarkCyan
}
