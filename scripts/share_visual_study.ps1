$ErrorActionPreference = 'Stop'
$studyWorkspace = Split-Path -Parent $PSScriptRoot
$studyData = 'D:\meongtamjeong_research\human_visual_study_v1'
$studyPython = 'C:\Users\sally\anaconda3\envs\dog-rag\python.exe'
$studyTunnel = 'D:\meongtamjeong_research\tools\cloudflared.exe'
$studyShare = Join-Path $studyData 'sharing'
New-Item -ItemType Directory -Path $studyShare -Force | Out-Null
if (-not (Test-Path -LiteralPath $studyTunnel)) { throw 'cloudflared.exe is missing.' }
$studyExisting = Join-Path $studyShare 'session.json'
if (Test-Path -LiteralPath $studyExisting) {
    $studySession = Get-Content -LiteralPath $studyExisting -Raw | ConvertFrom-Json
    $studyProcess = Get-Process -Id $studySession.tunnel_pid -ErrorAction SilentlyContinue
    if ($studyProcess -and $studyProcess.ProcessName -eq 'cloudflared') {
        Write-Output ('Existing evaluation link: ' + $studySession.url)
        exit 0
    }
}
$studyListener = Get-NetTCPConnection -LocalPort 8766 -State Listen -ErrorAction SilentlyContinue
if (-not $studyListener) {
    $studyServer = Start-Process -FilePath $studyPython -ArgumentList '-u','-m','experiments.dog_domain.visual_study_server','--port','8766','--public-raters-only' -WorkingDirectory $studyWorkspace -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $studyShare 'server.log') -RedirectStandardError (Join-Path $studyShare 'server.err')
    $studyServerId = $studyServer.Id
} else {
    $studyServerId = $studyListener.OwningProcess
}
$studyStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$studyLog = Join-Path $studyShare ('tunnel-'+$studyStamp+'.err')
$studyProc = Start-Process -FilePath $studyTunnel -ArgumentList 'tunnel','--url','http://127.0.0.1:8766','--no-autoupdate','--protocol','http2' -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $studyShare ('tunnel-'+$studyStamp+'.log')) -RedirectStandardError $studyLog
$studyUrl = $null
for ($studyTry = 0; $studyTry -lt 25; $studyTry++) {
    Start-Sleep -Seconds 1
    if (Test-Path -LiteralPath $studyLog) {
        $studyMatch = [regex]::Match((Get-Content -LiteralPath $studyLog -Raw), 'https://[a-z0-9-]+\.trycloudflare\.com')
        if ($studyMatch.Success) { $studyUrl = $studyMatch.Value; break }
    }
    if ($studyProc.HasExited) { throw ('Tunnel stopped; check '+$studyLog) }
}
if (-not $studyUrl) { throw ('Tunnel URL not ready; check '+$studyLog) }
@{url=$studyUrl;tunnel_pid=$studyProc.Id;server_pid=$studyServerId;created=(Get-Date).ToString('o');log=$studyLog} | ConvertTo-Json | Set-Content -LiteralPath $studyExisting -Encoding UTF8
$studyAccess = Get-Content -LiteralPath (Join-Path $studyData 'access.json') -Raw | ConvertFrom-Json
$studyRaters = @($studyAccess.raters.PSObject.Properties.Name | Where-Object { $_ -ne 'R1' })
$studyLines = @($studyRaters | ForEach-Object { $_+': '+$studyUrl+'/?key='+$studyAccess.raters.$_ })
# Link files stay outside the repository. Never publish the admin capability.
$studyLines | Set-Content -LiteralPath (Join-Path $studyShare 'FRIEND_LINKS.txt') -Encoding UTF8
foreach ($studyRater in $studyRaters) {
    ('[InternetShortcut]'+"`r`n"+'URL='+$studyUrl+'/?key='+$studyAccess.raters.$studyRater+"`r`n") | Set-Content -LiteralPath (Join-Path $studyShare ($studyRater+'.url')) -Encoding UTF8
}
Write-Output ('Evaluation sharing ready: '+$studyUrl)
