param([switch]$Lan)
$studyWorkspace = Split-Path -Parent $PSScriptRoot
$studyData = 'D:\meongtamjeong_research\human_visual_study_v1'
$studyPython = 'C:\Users\sally\anaconda3\envs\dog-rag\python.exe'
$studyBind = if ($Lan) { '0.0.0.0' } else { '127.0.0.1' }
$studyListener = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
if (-not $studyListener) {
    Start-Process -FilePath $studyPython -ArgumentList '-u','-m','experiments.dog_domain.visual_study_server','--host',$studyBind -WorkingDirectory $studyWorkspace -WindowStyle Hidden -RedirectStandardOutput (Join-Path $studyData 'server.log') -RedirectStandardError (Join-Path $studyData 'server.err')
}
$studyAccess = Get-Content -LiteralPath (Join-Path $studyData 'access.json') -Raw -Encoding UTF8 | ConvertFrom-Json
Start-Process ('http://localhost:8765/#' + $studyAccess.admin)
if ($Lan) { Write-Output 'LAN sharing requires a listener on 0.0.0.0; an already running localhost server is not restarted automatically.' }
