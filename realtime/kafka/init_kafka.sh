#!/usr/bin/env bash
# Golden rule 1: format the KRaft storage ONLY ONCE (first session). Reformatting erases the cluster metadata.
set -e
source "$(dirname "$0")/env.sh"
CFG="$KAFKA_HOME/config/server.properties"
if [ -f "$HERE/cluster_id.txt" ] && [ -f "$KAFKA_DATA/meta.properties" ]; then
  echo "Already formatted (cluster id $(cat "$HERE/cluster_id.txt")). Do NOT format again. Use start_kafka.sh."
  exit 0
fi
mkdir -p "$KAFKA_DATA"
# log.dirs -> a folder in your home (like the teacher's C:/kafka_4.3.1/kraft-data)
sed -i "s#^log.dirs=.*#log.dirs=${KAFKA_DATA}#" "$CFG"
grep -n "^log.dirs\|^process.roles\|^listeners\|^advertised.listeners" "$CFG"
ID=$("$KAFKA_HOME/bin/kafka-storage.sh" random-uuid)
echo "$ID" > "$HERE/cluster_id.txt"
"$KAFKA_HOME/bin/kafka-storage.sh" format --standalone -t "$ID" -c "$CFG"
echo "Formatted. Cluster id saved in realtime/kafka/cluster_id.txt (keep it with the project scripts - golden rule 4)."
echo "Next: bash realtime/kafka/start_kafka.sh   (in its own terminal), then bash realtime/kafka/create_topics.sh"
