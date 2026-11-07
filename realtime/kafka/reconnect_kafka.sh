#!/usr/bin/env bash
# MODUL_1 — reconnect after a disconnect, WITHOUT formatting again.
# 1) stop leftovers  2) check port 9092 is free  3) start the broker  4) check topics  5) consumers resume from
# their committed offsets (same group id) - nothing to do in the Python scripts, just start them again.
source "$(dirname "$0")/env.sh"
"$KAFKA_HOME/bin/kafka-server-stop.sh" 2>/dev/null; sleep 3
pkill -f kafka.Kafka 2>/dev/null; sleep 2
if ss -ltn | grep -q ":9092 "; then echo "port 9092 still busy:"; ss -ltnp | grep 9092; exit 1; fi
[ -f "$KAFKA_DATA/meta.properties" ] || { echo "storage not formatted: run init_kafka.sh once"; exit 1; }
nohup "$KAFKA_HOME/bin/kafka-server-start.sh" "$KAFKA_HOME/config/server.properties" > "$HERE/kafka.log" 2>&1 &
for i in $(seq 1 30); do "$KAFKA_HOME/bin/kafka-topics.sh" --list --bootstrap-server "$BOOTSTRAP" >/dev/null 2>&1 && break; sleep 1; done
echo "broker back. topics:"; "$KAFKA_HOME/bin/kafka-topics.sh" --list --bootstrap-server "$BOOTSTRAP"
echo "now restart the stopped producers/consumers: they resume from their committed offsets"
