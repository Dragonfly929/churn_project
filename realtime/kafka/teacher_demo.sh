#!/usr/bin/env bash
# Run the teacher's basic producer/consumer ONCE (plain Kafka round trip before the churn pipeline).
source "$(dirname "$0")/env.sh"
"$KAFKA_HOME/bin/kafka-topics.sh" --list --bootstrap-server "$BOOTSTRAP" | grep -qx system-metrics || \
  "$KAFKA_HOME/bin/kafka-topics.sh" --create --topic system-metrics --partitions 1 --replication-factor 1 --bootstrap-server "$BOOTSTRAP"
echo "Terminal A: python realtime/teacher/main_KAFKACONSUMER.py"
echo "Terminal B: python realtime/teacher/main_KAFKAPRODUSER.py      (Ctrl+C to stop)"
echo "Optional  : streamlit run realtime/teacher/app_river.py        (the teacher's River/RTAP app)"
