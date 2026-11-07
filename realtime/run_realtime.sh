#!/usr/bin/env bash
# Starts the whole real-time pipeline in the right order (golden rule 3), each service in the
# background with its own log in realtime/logs/. For the class demo you may instead open one
# terminal per command (printed below) - same effect, easier to show.
#
#   bash realtime/run_realtime.sh                 # with Kafka (start Kafka first: bash realtime/kafka/start_kafka.sh)
#   bash realtime/run_realtime.sh --bus file      # FALLBACK without Kafka
#   SECONDS_PER_DAY=1 DAYS=30 bash realtime/run_realtime.sh --bus file
#   bash realtime/run_realtime.sh --no-spark      # skip Spark (less RAM)
#   bash realtime/stop_realtime.sh                # stop everything
set -e
cd "$(dirname "$0")/.."
BUS=kafka; SPARK=1
while [ $# -gt 0 ]; do
  case "$1" in
    --bus) BUS="$2"; shift ;;
    --bus=*) BUS="${1#*=}" ;;
    --no-spark) SPARK=0 ;;
  esac
  shift
done
SPD=${SECONDS_PER_DAY:-2}; DAYS=${DAYS:-90}
PY=${PYTHON:-python}
mkdir -p realtime/logs realtime/output realtime/state
rm -rf realtime/output/* realtime/state/*.json realtime/bus
echo "{\"capacity\": 40, \"budget\": 200000, \"pause\": false, \"overrides\": {}}" > realtime/state/control.json
$PY - <<PYEOF
import sys; sys.path.insert(0, ".")
from realtime.bus import new_run, get_bus
info = new_run(seconds_per_day=$SPD, delay=25, days=$DAYS)
print("run", info)
if "$BUS" == "kafka":
    print("topics created:", get_bus("kafka").ensure_topics(partitions=3))
PYEOF
echo "== bus=$BUS  ($SPD s per bank day, $DAYS days)"
start() { echo "  starting $1"; nohup $PY "${@:2}" > "realtime/logs/$1.log" 2>&1 & echo $! >> realtime/state/pids; }
start scorer  realtime/alert_churn_MD.py --bus $BUS
start rtap    realtime/rtap_churn_MD.py --bus $BUS
start world   realtime/world_churn_MD.py --bus $BUS
start river   realtime/river_churn_MD.py --bus $BUS
[ "$SPARK" = 1 ] && start spark realtime/spark_churn_MD.py --bus $BUS
echo "  waiting for the scorer to load the DWH state..."
for i in $(seq 1 60); do grep -q "ready" realtime/logs/scorer.log 2>/dev/null && break; sleep 1; done
start prod_mobile_app     realtime/kafka_churn_MD.py --source mobile_app --bus $BUS
start prod_card_branch    realtime/kafka_churn_MD.py --source card_branch --bus $BUS
start prod_contact_center realtime/kafka_churn_MD.py --source contact_center --bus $BUS
echo "== all started. Dashboard:  streamlit run realtime/rtap_app.py"
echo "== logs: tail -f realtime/logs/scorer.log      stop: bash realtime/stop_realtime.sh"
