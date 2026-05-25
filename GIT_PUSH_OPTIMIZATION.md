# Git Push Performance Optimization Guide

## Status Summary
- **Current Branch**: main
- **Status**: 1 commit ahead of origin/main
- **Remote**: https://github.com/YoukaiKouhai/Optimal-LV-Placement-ML
- **Branch Tracking**: Correct (tracking origin/main)

## Fixes Applied ✅

The following optimizations have been applied to your git configuration:

### 1. Increased Post-Buffer Size
```bash
git config --global http.postBuffer 524288000
```
**Why**: Prevents git from staging large commits in chunks. Default is too small for repositories with many files.

### 2. Increased HTTP Timeout
```bash
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999
```
**Why**: Prevents premature disconnection on slow/intermittent connections.

### 3. Disabled Compression
```bash
git config --global core.compression 0
```
**Why**: Trades bandwidth for speed - useful for slow network connections where CPU is the bottleneck.

---

## Next Steps - Try These Commands in Order

### Step 1: Test Remote Connection
```bash
cd "c:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML"
git fetch origin -v
```
Expected output: Should show refs and commits from remote (may take a moment).

### Step 2: Attempt Push with Progress
```bash
cd "c:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML"
git push -v origin main
```
This will show:
- ✅ Success: `Pushing to https://...` followed by commit updates
- ⏳ Slow: Takes > 30 seconds (proceed to advanced fixes)
- ❌ Hangs: Appears stuck (proceed to authentication fixes)

### Step 3: If Push is Still Slow (>30 seconds)

**Check for large files first:**
```bash
git ls-tree -r -l HEAD | sort -k 4 -rn | head -20
```
This shows the 20 largest files in your repo. If any .pt files (PyTorch models) or .json files are > 100MB, see "Handle Large Files" section.

### Step 4: If Push Hangs or Asks for Password

**Clear cached credentials (Windows):**
```bash
git credential-manager-core erase https://github.com
```

Then try:
```bash
git push -v origin main
```
Git will prompt for credentials fresh, which often fixes hanging issues.

---

## Alternative: Use SSH Instead of HTTPS (Often Faster & More Reliable)

### Step A: Check SSH Setup
```bash
Test-Path $env:USERPROFILE\.ssh\id_rsa
```

### Step B: If SSH key exists, switch remote to SSH
```bash
git remote set-url origin git@github.com:YoukaiKouhai/Optimal-LV-Placement-ML.git
```

### Step C: Try push via SSH
```bash
git push -v origin main
```

---

## Handle Large Files (If Needed)

If you have model files (.pt), checkpoint files, or large data files:

### Check total size of pending push:
```bash
$files = git diff-tree --no-commit-id --name-only -r HEAD
$total = 0
foreach ($f in $files) {
    if (Test-Path $f) {
        $size = (Get-Item $f).Length
        $total += $size
        if ($size -gt 5MB) { 
            Write-Host "$f: $([math]::Round($size/1MB,1))MB"
        }
    }
}
Write-Host "Total size: $([math]::Round($total/1MB,1))MB"
```

**If > 500MB:**
- Consider using Git LFS (Git Large File Storage)
- Split into multiple smaller commits
- Remove non-essential files (model checkpoints, temp data)

### Set up Git LFS (if needed):
```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push origin main
```

---

## Network/Connection Debugging

If push still hangs after trying above steps:

### Monitor connection during push:
1. Open Task Manager → Performance tab
2. Start git push
3. Watch network activity - if 0 Mbps, connection issue
4. Check if you're on stable network (wired preferred over WiFi)

### Try with explicit timeout:
```bash
$env:GIT_TRACE=1
git push -v origin main
```
This shows detailed debug output to identify where it's hanging.

---

## Last Resort Options

### Option 1: Restart Windows Credential Manager
```powershell
# Stop credential manager
Get-Process lsass -ErrorAction SilentlyContinue | Stop-Process -Force

# Then retry push - Windows will restart services
git push origin main
```

### Option 2: Reinstall Git
- Download latest from https://git-scm.com/download/win
- Uninstall current version (Settings → Apps & features)
- Reinstall with default options
- Retry push

### Option 3: Use GitHub CLI instead
```bash
# Install: https://cli.github.com/
gh auth login
gh repo sync
```

---

## Verification

After successful push, verify:
```bash
git status
# Should show: "Your branch is up to date with 'origin/main'"

git log --oneline -5
# Latest commit should show (HEAD -> main)
```

---

## Questions to Ask If Still Stuck

- Network: Is connection stable? (test with ping google.com)
- Size: How large is the commit? (run the size check above)
- Auth: Are you being asked for credentials?
- Time: How long does it take to fail/hang? (exact seconds)
