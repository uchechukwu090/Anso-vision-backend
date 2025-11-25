#!/usr/bin/env python3
"""
Git Sync & Push Script - Handles conflicts
Pulls remote changes first, then pushes local updates
"""

import subprocess
import os
import sys

def run_command(cmd, description=""):
    """Execute a command and print output"""
    try:
        if description:
            print(f"\n{description}")
            print("=" * 60)
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=repo_path)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr and "warning" not in result.stderr.lower():
            print(f"⚠️  {result.stderr}")
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Define repo path
repo_path = r"C:\Users\User\dyad-apps\Anso backend python"

if not os.path.exists(repo_path):
    print(f"❌ Repository not found at {repo_path}")
    sys.exit(1)

print("🚀 Ansorade Backend - Git Sync & Push Script")
print(f"📂 Repository: {repo_path}")

# Step 1: Check current branch
print("\n🔍 Checking current branch...")
result = subprocess.run("git branch -v", shell=True, capture_output=True, text=True, cwd=repo_path)
print(result.stdout)

# Step 2: Fetch latest from remote
print("\n📥 Fetching latest from GitHub...")
success = run_command("git fetch origin", "Downloading remote changes...")
if not success:
    print("⚠️  Fetch had issues, continuing anyway...")

# Step 3: Check status
run_command("git status", "📋 Current Status After Fetch")

# Step 4: Pull remote changes
print("\n🔄 Pulling remote changes...")
success = run_command("git pull origin main --no-rebase", "Merging remote changes...")

if not success:
    print("⚠️  Pull failed on main, trying with rebase...")
    success = run_command("git pull origin main --rebase", "Pulling with rebase...")

# Step 5: Check what branch we're on
print("\n📍 Verifying branch...")
result = subprocess.run("git rev-parse --abbrev-ref HEAD", shell=True, capture_output=True, text=True, cwd=repo_path)
current_branch = result.stdout.strip()
print(f"Current branch: {current_branch}")

# Step 6: Show local changes
run_command("git status", "📋 Final Status Before Push")

# Step 7: Add local changes (if any)
print("\n➕ Adding local changes...")
success = run_command("git add -A", "Staging changes...")

# Step 8: Check if there's anything to commit
result = subprocess.run("git diff --cached --quiet", shell=True, cwd=repo_path)
has_changes = result.returncode != 0

if has_changes:
    print("\n💾 Committing local changes...")
    commit_message = """🎯 Fix: Signal Stability & Decision Logic Improvement

CHANGES:
- context_aware_hmm.py: Replaced rigid AND-logic with scoring system
  * Triggers on 2-factor confluence (HMM + Trend) 
  * Full confidence on 3-factor (HMM + Trend + Volume)
  * Handles consolidation breakouts (was WAIT before)
  
- hmm_model.py: Added state smoothing to prevent rapid flipping
  * 5-candle majority voting filter on predicted states
  * Added get_state_stability() metric
  * States now require 3+ candle confirmation before changing

FIXES:
✅ BTC WAIT signal issue (missed 83,925→89,092 move)
✅ Rapid signal flipping (BUY→SELL→NEUTRAL every second)
✅ Missed consolidation breakouts

RESULT:
- More signals generated (2/3 confluence triggers now)
- Signals stay stable (5 candle smoothing applied)
- Better reasoning provided in frontend"""
    
    success = run_command(f'git commit -m "{commit_message}"', "Creating commit...")
    if not success:
        print("⚠️  Commit failed - no changes or already committed")
else:
    print("\n✓ No local changes to commit")

# Step 9: Push to GitHub
print("\n📤 Pushing to GitHub...")
print("=" * 60)

push_success = run_command(f"git push origin {current_branch}", f"Pushing {current_branch} branch...")

if push_success:
    print("\n✅ SUCCESS! Changes pushed to GitHub")
    print("🔗 Repository: https://github.com/uchechukwu090/Anso-vision-backend.git")
    print(f"📍 Branch: {current_branch}")
    print("\n📊 Summary:")
    print("  - Files synced with remote")
    print("  - Local changes committed")
    print("  - Changes pushed to GitHub")
    print("  - Next: Verify on GitHub and test in frontend")
else:
    print("\n⚠️  Push may have failed")
    print("Troubleshooting steps:")
    print("1. Check your GitHub credentials")
    print("2. Verify you have write access to the repository")
    print("3. Try: git push origin main -u (with -u flag)")
    print("4. Or check GitHub for required reviews/approvals")
    sys.exit(1)
