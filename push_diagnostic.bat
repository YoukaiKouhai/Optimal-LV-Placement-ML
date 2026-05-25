@echo off
REM Git Push Optimization and Troubleshooting Script
REM Run this in PowerShell or Command Prompt as Administrator

setlocal enabledelayedexpansion

cd /d "c:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML"

echo.
echo ===== GIT PUSH DIAGNOSTIC AND FIX SCRIPT =====
echo.
echo Step 1: Checking git status...
timeout /t 2 /nobreak
git status
echo.

echo Step 2: Clearing credentials...
timeout /t 2 /nobreak
git credential-manager-core erase https://github.com
echo Credentials cleared.
echo.

echo Step 3: Checking file sizes in commit...
timeout /t 2 /nobreak
PowerShell -Command ^
  "$files = git diff-tree --no-commit-id --name-only -r HEAD; $total = 0; foreach ($f in $files) { if (Test-Path $f) { $size = (Get-Item $f).Length; $total += $size; if ($size -gt 5MB) { Write-Host \"$f: $([math]::Round($size/1MB,1))MB\" } } }; Write-Host \"Total size: $([math]::Round($total/1MB,1))MB\""
echo.

echo Step 4: Attempting push with GIT_TRACE enabled...
echo This will show detailed connection information.
timeout /t 2 /nobreak
set GIT_TRACE=1
git push -v origin main

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Push completed!
    git status
) else (
    echo.
    echo [FAILED] Push encountered an error.
    echo Troubleshooting suggestions:
    echo.
    echo 1. Check your internet connection
    echo 2. Verify GitHub is not down: https://www.githubstatus.com
    echo 3. If stuck on password, press Ctrl+C and try:
    echo    git push -v origin main --progress
    echo 4. Try SSH instead:
    echo    git remote set-url origin git@github.com:YoukaiKouhai/Optimal-LV-Placement-ML.git
    echo    git push -v origin main
    echo.
)

pause
