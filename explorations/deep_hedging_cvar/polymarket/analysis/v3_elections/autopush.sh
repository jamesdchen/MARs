#!/bin/bash
# Hourly (at :07) commit + push of w3's folders only: compacted recorder files, reward snapshots,
# selection, scripts, and the crypto w3 ledger. Merge pulls (no rebase), never force.
PM=/home/user/MARs/explorations/deep_hedging_cvar/polymarket
B=claude/eloquent-feynman-n78cn1
while true; do
  now=$(date -u +%s); next=$(( (now/3600+1)*3600 + 420 )); sleep $(( next - now ))
  (
    flock 9
    cd $PM || exit
    cp analysis/v3_elections/rec/status.json analysis/v3_elections/rec/status_last.json 2>/dev/null
    git add analysis/v3_elections/*.py analysis/v3_elections/*.sh analysis/v3_elections/*.md analysis/v3_elections/.gitignore 2>/dev/null
    git add -f analysis/v3_elections/sel analysis/v3_elections/rec/status_last.json 2>/dev/null
    for d in books reconcile market_qc markets trades rewards coverage; do
      [ -d analysis/v3_elections/rec/$d ] && find analysis/v3_elections/rec/$d -type f \( -name '*.parquet' -o -name '*.jsonl' \) -not -path '*/_parts/*' -print0 | xargs -0 -r git add -f
    done
    for f in train/workers/w3/*.jsonl train/workers/w3/*.py train/workers/w3/*.md; do [ -e "$f" ] && git add "$f"; done
    git commit -qm "w3: hourly election recording push $(date -u +%FT%H:%MZ)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MAy8XXopCJb7KbJowxeVQY" && \
    for i in 1 2 3 4 5; do git pull -q --no-rebase origin $B && git push -q origin HEAD:$B && break; sleep $((2**i)); done
    echo "$(date -u +%FT%TZ) pushed $(git log -1 --format=%h)"
  ) 9>/tmp/w3_git.lock
done
