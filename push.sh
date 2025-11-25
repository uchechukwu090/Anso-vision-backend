#!/bin/bash
# Script to push Ansorade backend changes to GitHub

cd "C:\Users\User\dyad-apps\Anso backend python" || exit 1

echo "📝 Git Status Before Commit:"
git status

echo ""
echo "🔄 Adding all changes..."
git add -A

echo ""
echo "💾 Committing changes..."
git commit -m "🎯 Fix: Signal Stability & Decision Logic

- Updated context_aware_hmm.py with scoring-based decision logic
  - Removed rigid AND-logic that required perfect confluence
  - Now triggers on HMM + Trend (2/3 factors)
  - Full signal on HMM + Trend + Volume (3/3 factors)

- Updated hmm_model.py with state smoothing
  - Added 5-candle majority voting filter
  - Prevents rapid state flipping (e.g., BUY->SELL->NEUTRAL every second)
  - Added get_state_stability() to track state confidence

- Result: BTC WAIT issue fixed
  - Now signals BUY/SELL on 2-factor confluence vs waiting for 3
  - State changes require 3+ candle confirmation (was 1 before)
  - Handles consolidation breakouts that were previously ignored

Addresses: Missed signals on BTC 83,925->89,092 move
Fixes: Rapid signal flipping causing whipsaws"

echo ""
echo "📤 Pushing to GitHub..."
git push origin main || git push origin master

echo ""
echo "✅ Done! Changes pushed to GitHub"
echo "Repository: https://github.com/uchechukwu090/Anso-vision-backend.git"
