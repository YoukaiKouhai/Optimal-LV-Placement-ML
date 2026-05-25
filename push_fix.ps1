#!/usr/bin/env pwsh
# Git Push Optimization and Troubleshooting Script for PowerShell
# Run this with: powershell -ExecutionPolicy Bypass -File push_fix.ps1

Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  GIT PUSH OPTIMIZATION & TROUBLESHOOTING SCRIPT                ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$repoPath = "c:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML"

if (-not (Test-Path $repoPath)) {
    Write-Host "ERROR: Repository path not found: $repoPath" -ForegroundColor Red
    exit 1
}

Set-Location $repoPath

# ============================================================================
# STEP 1: Git Status Check
# ============================================================================
Write-Host "📋 STEP 1: Current Git Status" -ForegroundColor Yellow
Write-Host "─" * 60
git status
Write-Host ""
Start-Sleep -Seconds 2

# ============================================================================
# STEP 2: Configuration Check
# ============================================================================
Write-Host "⚙️  STEP 2: Git Configuration" -ForegroundColor Yellow
Write-Host "─" * 60
Write-Host "Post Buffer Size:" (git config --global http.postBuffer) -ForegroundColor Green
Write-Host "Compression:" (git config --global core.compression) -ForegroundColor Green
Write-Host ""
Start-Sleep -Seconds 2

# ============================================================================
# STEP 3: File Size Analysis
# ============================================================================
Write-Host "📊 STEP 3: Commit File Size Analysis" -ForegroundColor Yellow
Write-Host "─" * 60

$files = git diff-tree --no-commit-id --name-only -r HEAD
$largeFiles = @()
$totalSize = 0

foreach ($file in $files) {
    if (Test-Path $file) {
        $size = (Get-Item $file).Length
        $totalSize += $size
        $sizeMB = [math]::Round($size / 1MB, 2)
        
        if ($sizeMB -gt 1) {
            $largeFiles += @{File = $file; Size = $sizeMB}
            Write-Host "$file : $sizeMB MB" -ForegroundColor Magenta
        }
    }
}

$totalMB = [math]::Round($totalSize / 1MB, 2)
Write-Host ""
Write-Host "Total commit size: $totalMB MB" -ForegroundColor Cyan
Write-Host ""

if ($totalMB -gt 500) {
    Write-Host "⚠️  WARNING: Commit is large (>500 MB)" -ForegroundColor Red
    Write-Host "   This may be slow to push. Consider:"
    Write-Host "   1. Reducing commit size"
    Write-Host "   2. Using Git LFS for large files"
    Write-Host "   3. Splitting into multiple commits"
    Write-Host ""
}

Start-Sleep -Seconds 2

# ============================================================================
# STEP 4: Credential Reset
# ============================================================================
Write-Host "🔐 STEP 4: Clearing Cached Credentials" -ForegroundColor Yellow
Write-Host "─" * 60
git credential-manager-core erase https://github.com 2>&1 | Where-Object { $_ -ne "" } | ForEach-Object { Write-Host $_ }
Write-Host "✓ Credentials cleared" -ForegroundColor Green
Write-Host ""
Start-Sleep -Seconds 2

# ============================================================================
# STEP 5: Remote Configuration
# ============================================================================
Write-Host "🌐 STEP 5: Remote Configuration" -ForegroundColor Yellow
Write-Host "─" * 60
$remoteUrl = git config --get remote.origin.url
Write-Host "Current remote URL:" -ForegroundColor Cyan
Write-Host "  $remoteUrl" -ForegroundColor Green
Write-Host ""

Write-Host "Protocol options:" -ForegroundColor Cyan
Write-Host "  1. Keep HTTPS (current)"
Write-Host "  2. Switch to SSH (potentially faster)"
Write-Host ""

$choice = Read-Host "Enter choice (1 or 2) or press Enter to continue with HTTPS"

if ($choice -eq "2") {
    Write-Host ""
    Write-Host "Switching to SSH..." -ForegroundColor Yellow
    git remote set-url origin git@github.com:YoukaiKouhai/Optimal-LV-Placement-ML.git
    Write-Host "✓ Remote updated to SSH" -ForegroundColor Green
    Write-Host ""
}

# ============================================================================
# STEP 6: Attempt Push with Enhanced Output
# ============================================================================
Write-Host "🚀 STEP 6: Attempting Push" -ForegroundColor Yellow
Write-Host "─" * 60
Write-Host "Starting push... (this may take a moment)" -ForegroundColor Cyan
Write-Host ""

$pushStart = Get-Date

# Enable Git trace for debugging
$env:GIT_TRACE = 1
$env:GIT_CURL_VERBOSE = 1

$pushOutput = git push -v origin main 2>&1
$pushEnd = Get-Date
$pushDuration = ($pushEnd - $pushStart).TotalSeconds

Write-Host $pushOutput
Write-Host ""

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ PUSH SUCCESSFUL!" -ForegroundColor Green -BackgroundColor Black
    Write-Host "   Duration: $pushDuration seconds" -ForegroundColor Green
    Write-Host ""
    
    # Verify push
    Write-Host "Verifying..." -ForegroundColor Cyan
    git status
} else {
    Write-Host "❌ PUSH FAILED" -ForegroundColor Red -BackgroundColor Black
    Write-Host "   Duration: $pushDuration seconds" -ForegroundColor Red
    Write-Host "   Exit code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host ""
    
    if ($pushDuration -lt 5) {
        Write-Host "Quick failure - likely authentication issue" -ForegroundColor Yellow
        Write-Host "Try logging in manually with: gh auth login" -ForegroundColor Cyan
    } elseif ($pushDuration -gt 60) {
        Write-Host "Slow failure - likely network or size issue" -ForegroundColor Yellow
        Write-Host "Check internet connection or reduce commit size" -ForegroundColor Cyan
    }
}

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  TROUBLESHOOTING NEXT STEPS                                    ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "If still having issues:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1️⃣  Check GitHub status: https://www.githubstatus.com"
Write-Host ""
Write-Host "2️⃣  Try using GitHub CLI:"
Write-Host "    gh auth login"
Write-Host "    gh repo sync"
Write-Host ""
Write-Host "3️⃣  Monitor connection during push:"
Write-Host "    Open Task Manager > Performance tab"
Write-Host "    Watch Network while running push"
Write-Host ""
Write-Host "4️⃣  Use SSH (may be faster):"
Write-Host "    git remote set-url origin git@github.com:YoukaiKouhai/Optimal-LV-Placement-ML.git"
Write-Host "    git push -v origin main"
Write-Host ""
Write-Host "5️⃣  Split large commits:"
Write-Host "    git reset HEAD~1  (undo last commit)"
Write-Host "    git add <smaller-subset-of-files>"
Write-Host "    git commit -m 'Part 1: ...'"
Write-Host "    git push origin main"
Write-Host ""
