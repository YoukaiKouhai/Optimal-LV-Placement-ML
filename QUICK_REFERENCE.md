# Git Push Troubleshooting Quick Reference

## 🚀 Quick Fix (Try This First)

```bash
# 1. Clear credentials
git credential-manager-core erase https://github.com

# 2. Try push again
cd "c:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML"
git push origin main
```

---

## 📋 Diagnostic Checklist

- [ ] Run `git status` - shows how many commits ahead you are
- [ ] Run `git fetch origin` - updates local knowledge of remote
- [ ] Check GitHub website - verify your branch appears there
- [ ] Test internet - `ping google.com` should respond
- [ ] Check GitHub status - https://www.githubstatus.com
- [ ] Look for large files - `git ls-tree -r -l HEAD | sort -k4 -rn | head`

---

## ⚡ What We Already Applied

✅ `git config --global http.postBuffer 524288000` - Large file buffer
✅ `git config --global http.lowSpeedLimit 0` - Prevent timeout  
✅ `git config --global http.lowSpeedTime 999999` - Extended timeout
✅ `git config --global core.compression 0` - Faster compression

---

## 🔧 Common Fixes (In Order)

### Fix #1: Credential Issue (Most Common)
```bash
git credential-manager-core erase https://github.com
git push origin main
```

### Fix #2: Try SSH Instead (Often Faster)
```bash
git remote set-url origin git@github.com:YoukaiKouhai/Optimal-LV-Placement-ML.git
git push -v origin main
```

### Fix #3: Check Connection
```bash
# Test connectivity
curl -I https://github.com

# Or in PowerShell:
Test-NetConnection github.com -Port 443
```

### Fix #4: Small Commit in Pieces
```bash
# Undo last commit
git reset HEAD~1

# Stage and push smaller parts
git add <some-files>
git commit -m "Part 1: ..."
git push origin main

# Then repeat for remaining files
```

### Fix #5: Use GitHub CLI (Alternative)
```bash
# Download from https://cli.github.com/
gh auth login
gh repo sync
```

---

## 🛠️ Run These Scripts

### Option A: PowerShell (Recommended - More Detailed)
```powershell
powershell -ExecutionPolicy Bypass -File push_fix.ps1
```

### Option B: Batch Script
```cmd
push_diagnostic.bat
```

---

## 📊 Check Commit Size

```powershell
# PowerShell
$files = git diff-tree --no-commit-id --name-only -r HEAD
$total = 0
foreach ($f in $files) {
    if (Test-Path $f) {
        $size = (Get-Item $f).Length
        $total += $size
        if ($size -gt 10MB) { 
            Write-Host "$f: $('{0:N1}' -f ($size/1MB))MB" 
        }
    }
}
Write-Host "Total: $('{0:N1}' -f ($total/1MB))MB"
```

---

## 📞 Still Stuck? Provide These Details

- **Size**: Total commit size in MB
- **Time**: How long does it take to fail? (seconds)
- **Error**: Exact error message shown
- **Network**: Wired or WiFi? Speed?
- **Auth**: Password prompt shown or no?
- **History**: Can you push small changes normally?

---

## ✅ Verify Success

```bash
git status
# Should show: "Your branch is up to date with 'origin/main'"
```

---

## 🔄 Alternative: Force Fresh Clone

If everything else fails:
```bash
cd ..
mv "Optimal-LV-Placement-ML" "Optimal-LV-Placement-ML.backup"
git clone https://github.com/YoukaiKouhai/Optimal-LV-Placement-ML.git
cd Optimal-LV-Placement-ML
# Copy your changes back from the backup folder
```

---

## 🌐 Network Optimization

If on slow network, enable minimal compression:
```bash
git config --global http.version HTTP/1.1
git config --global http.keepalive false
```

To revert:
```bash
git config --global http.version
git config --global http.keepalive
```
