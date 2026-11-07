#!/usr/bin/env bash
cd "$(dirname "$0")/.."
[ -f realtime/state/pids ] && xargs -r kill < realtime/state/pids 2>/dev/null; rm -f realtime/state/pids
pkill -f "realtime/(alert|rtap|world|river|spark|kafka)_churn_MD.py" 2>/dev/null
echo "stopped"
