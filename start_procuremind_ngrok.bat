@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT_DIR=%~dp0"
set "BACKEND_DIR=%ROOT_DIR%ProcureMind backend"
set "FRONTEND_DIR=%ROOT_DIR%Procuremind frontend"
set "NGROK_TOKEN_FILE=%ROOT_DIR%ngrok.txt"
set "NGROK_CFG=%TEMP%\procuremind-ngrok.yml"
set "NGROK_API=http://127.0.0.1:4041/api/tunnels"

echo ================================================
echo ProcureMind Launcher (Frontend + Backend + 2 ngrok)
echo ================================================
echo.

if not exist "%BACKEND_DIR%\main.py" (
  echo [ERROR] Backend folder not found at "%BACKEND_DIR%"
  exit /b 1
)

if not exist "%FRONTEND_DIR%\package.json" (
  echo [ERROR] Frontend folder not found at "%FRONTEND_DIR%"
  exit /b 1
)

where ngrok >nul 2>nul
if errorlevel 1 (
  echo [ERROR] ngrok is not in PATH. Install it and ensure `ngrok` runs in terminal.
  exit /b 1
)

echo.
echo [1/7] Configuring ngrok auth token...
if not exist "%NGROK_TOKEN_FILE%" (
  echo [ERROR] Token file not found: "%NGROK_TOKEN_FILE%"
  echo         Create ngrok.txt with your ngrok authtoken.
  exit /b 1
)

set "NGROK_TOKEN="
for /f "usebackq delims=" %%t in ("%NGROK_TOKEN_FILE%") do (
  if not defined NGROK_TOKEN set "NGROK_TOKEN=%%t"
)

if not defined NGROK_TOKEN (
  echo [ERROR] ngrok.txt is empty. Put your authtoken in the first line.
  exit /b 1
)

ngrok config add-authtoken !NGROK_TOKEN! >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Failed to configure ngrok authtoken.
  echo         Validate the token value in ngrok.txt.
  exit /b 1
)

echo [OK] ngrok authtoken configured.

echo [2/7] Creating ngrok multi-tunnel config...
(
  echo version: "2"
  echo authtoken: !NGROK_TOKEN!
  echo web_addr: 127.0.0.1:4041
  echo tunnels:
  echo   frontend:
  echo     proto: http
  echo     addr: 5173
  echo   backend:
  echo     proto: http
  echo     addr: 8000
) > "%NGROK_CFG%"

echo [3/7] Starting ngrok with frontend and backend tunnels...
start "ProcureMind ngrok" cmd /k "ngrok start --all --config ""%NGROK_CFG%"""

echo [4/7] Waiting for ngrok tunnel URLs...
set "FRONTEND_NGROK_URL="
set "BACKEND_NGROK_URL="
for /l %%i in (1,1,35) do (
  for /f "usebackq delims=" %%u in (`powershell -NoProfile -Command "$t=(Invoke-RestMethod -Uri '%NGROK_API%' -ErrorAction SilentlyContinue).tunnels; if($t){($t | Where-Object { $_.name -eq 'frontend' } | Select-Object -First 1 -ExpandProperty public_url)}"`) do (
    set "FRONTEND_NGROK_URL=%%u"
  )

  for /f "usebackq delims=" %%u in (`powershell -NoProfile -Command "$t=(Invoke-RestMethod -Uri '%NGROK_API%' -ErrorAction SilentlyContinue).tunnels; if($t){($t | Where-Object { $_.name -eq 'backend' } | Select-Object -First 1 -ExpandProperty public_url)}"`) do (
    set "BACKEND_NGROK_URL=%%u"
  )

  if defined FRONTEND_NGROK_URL if defined BACKEND_NGROK_URL goto :got_urls
  timeout /t 2 >nul
)

echo [ERROR] Could not read ngrok tunnel URLs from %NGROK_API%
echo         Check the ngrok window for startup/auth errors.
goto :end

:got_urls
echo [OK] Frontend tunnel: !FRONTEND_NGROK_URL!
echo [OK] Backend tunnel : !BACKEND_NGROK_URL!

echo [5/7] Writing frontend .env with backend tunnel URL...
(
  echo # FastAPI backend URL
  echo VITE_API_URL=!BACKEND_NGROK_URL!
) > "%FRONTEND_DIR%\.env"

echo [6/7] Starting backend on port 8000 with CORS origin set to frontend tunnel...
start "ProcureMind Backend" cmd /k "cd /d ""%BACKEND_DIR%"" && if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat && set PROCUREMIND_ALLOWED_ORIGINS=!FRONTEND_NGROK_URL! && uvicorn procuremind.api:app --host 0.0.0.0 --port 8000"

timeout /t 2 >nul

echo [7/7] Starting frontend dev server on port 5173...
start "ProcureMind Frontend" cmd /k "cd /d ""%FRONTEND_DIR%"" && npm run dev -- --host 0.0.0.0 --port 5173"

echo.
echo Done.
echo - Public Frontend URL (share this): !FRONTEND_NGROK_URL!
echo - Public Backend URL: !BACKEND_NGROK_URL!
echo - Local Frontend: http://localhost:5173
echo - Local Backend:  http://localhost:8000/health
echo - ngrok API:      %NGROK_API%
echo.
echo Keep all three windows open: ngrok, backend, frontend.
echo If browser shows stale API endpoint, do a hard refresh.

:end
endlocal
