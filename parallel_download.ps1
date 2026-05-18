$outFile  = "C:\VideoUpscale\output\Mortal Kombat II_4K_upscaled.mp4"
$tmpDir   = "C:\VideoUpscale\output"
$total    = 19569961241
$done     = (Get-Item $outFile).Length
$url      = "http://localhost:{0}/Mortal%20Kombat%20II_4K_upscaled.mp4"
$ports    = 7777, 7778, 7779, 7780
$workers  = $ports.Count
$remain   = $total - $done
$chunk    = [math]::Ceiling($remain / $workers)

Write-Host "Resume from byte $done  ($([math]::Round($done/1GB,2)) GB)"
Write-Host "Remaining: $([math]::Round($remain/1GB,2)) GB  split into $workers chunks of ~$([math]::Round($chunk/1GB,2)) GB"

# Launch parallel chunk downloads
$jobs = @()
for ($i = 0; $i -lt $workers; $i++) {
    $start = $done + ($i * $chunk)
    $end   = [math]::Min($start + $chunk - 1, $total - 1)
    $tmp   = "$tmpDir\chunk_$i.tmp"
    $port  = $ports[$i]
    Write-Host "Job $i : bytes $start-$end  -> $tmp  (port $port)"
    $jobs += Start-Job -ScriptBlock {
        param($s,$e,$p,$t,$u)
        & curl.exe --range "$s-$e" ($u -f $p) -o $t --progress-bar 2>&1
    } -ArgumentList $start,$end,$port,$tmp,$url
}

Write-Host "`nAll $workers jobs started. Waiting for completion...`n"

# Progress loop
while ($true) {
    $running = $jobs | Where-Object { $_.State -eq 'Running' }
    if (-not $running) { break }
    $sizes = for ($i=0;$i -lt $workers;$i++) {
        $t = "$tmpDir\chunk_$i.tmp"
        if (Test-Path $t) { (Get-Item $t).Length } else { 0 }
    }
    $totalDl = ($sizes | Measure-Object -Sum).Sum
    $pct = [math]::Round(($done + $totalDl) / $total * 100, 1)
    $dlGB = [math]::Round($totalDl/1GB,2)
    Write-Host -NoNewline "`r[$pct%] chunks: $($sizes | ForEach-Object { [math]::Round($_/1MB,0)+'MB' } | Join-String -Separator ' | ')  total new: ${dlGB}GB   "
    Start-Sleep 5
}

Write-Host "`n`nAll chunks done. Assembling..."

# Append chunks in order
$fs = [System.IO.FileStream]::new($outFile, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write)
$buf = New-Object byte[] (8MB)
for ($i = 0; $i -lt $workers; $i++) {
    $tmp = "$tmpDir\chunk_$i.tmp"
    if (-not (Test-Path $tmp)) { Write-Error "Missing $tmp"; break }
    $cs = [System.IO.FileStream]::new($tmp, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read)
    $read = 0
    while (($read = $cs.Read($buf, 0, $buf.Length)) -gt 0) { $fs.Write($buf, 0, $read) }
    $cs.Close()
    Remove-Item $tmp
    Write-Host "Appended chunk $i"
}
$fs.Close()

$finalSize = (Get-Item $outFile).Length
Write-Host "`nFinal file: $([math]::Round($finalSize/1GB,2)) GB  (expected: $([math]::Round($total/1GB,2)) GB)"
if ($finalSize -eq $total) { Write-Host "SUCCESS - sizes match!" } else { Write-Host "WARNING - size mismatch!" }
