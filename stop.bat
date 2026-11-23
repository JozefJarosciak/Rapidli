@echo off
rem Stops Rapidli and the engine it started. It asks Rapidli to shut down; no process is stopped by name.
rem An engine started by hand (python app\eng.py start ...) is stopped with: python app\eng.py stop
powershell -NoProfile -Command "$u = 'http://127.0.0.1:11842/api/'; try { Invoke-RestMethod -Method Post -Uri ($u + 'shutdown') -Body '{}' -ContentType 'application/json' | Out-Null } catch { 'Rapidli is not running.'; exit }; foreach ($i in 1..60) { try { Invoke-RestMethod -Uri ($u + 'status') -TimeoutSec 2 | Out-Null; Start-Sleep -Milliseconds 500 } catch { Start-Sleep -Seconds 1; 'Rapidli and its engine are stopped.'; exit } }; 'Rapidli was asked to stop and is still at it; look again in a moment.'"
