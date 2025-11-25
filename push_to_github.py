#!/usr/bin/env python3
"""
Git Push Script for Ansorade Backend
Commits and pushes HMM signal fixes to GitHub
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

print("🚀 Ansorade Backend - Git Push Script")
print(f"📂 Repository: {repo_path}")

# Step 1: Check git status
run_command("git status", "📋 Current Git Status")

# Step 2: Add all changes
success = run_command("git add -A", "➕ Adding all changes...")
if not success:
    print("❌ Failed to add changes")
    sys.exit(1)

# Step 3: Commit changes
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

success = run_command(f'git commit -m "{commit_message}"', "💾 Committing changes...")
if not success:
    print("⚠️  Commit may have failed or nothing to commit")

# Step 4: Push to GitHub
print("\n📤 Pushing to GitHub...")
print("=" * 60)

# Try main branch first
push_success = run_command("git push origin main", "Attempting push to main branch...")

if not push_success:
    print("\n⚠️  Main branch push failed, trying master...")
    push_success = run_command("git push origin master", "Attempting push to master branch...")

if push_success:
    print("\n✅ SUCCESS! Changes pushed to GitHub")
    print("🔗 Repository: https://github.com/uchechukwu090/Anso-vision-backend.git")
    print("\n📊 Summary:")
    print("  - Files modified: context_aware_hmm.py, hmm_model.py")
    print("  - Signal logic: More triggers, more stable")
    print("  - Next: Verify in frontend on BTCUSD/EURUSD/etc")
else:
    print("\n❌ Push to GitHub failed")
    print("Try running: git push origin main (or master)")
    sys.exit(1)
