#!/usr/bin/env bash
# Step 0 / MODUL_0 — install Apache Kafka (same version as the teacher's guide) into ~/kafka
set -e
source "$(dirname "$0")/env.sh"
command -v java >/dev/null || { echo "Java missing: sudo apt install -y openjdk-17-jdk (or openjdk-21-jdk)"; exit 1; }
java -version 2>&1 | head -1
mkdir -p "$HOME/kafka" && cd "$HOME/kafka"
TGZ=kafka_${SCALA}-${KAFKA_VERSION}.tgz
if [ ! -d "$KAFKA_HOME" ]; then
  wget -q --show-progress "https://downloads.apache.org/kafka/${KAFKA_VERSION}/${TGZ}" \
    || wget -q --show-progress "https://archive.apache.org/dist/kafka/${KAFKA_VERSION}/${TGZ}"
  tar -xzf "$TGZ" && rm "$TGZ"
fi
echo "Kafka installed in $KAFKA_HOME"
ls "$KAFKA_HOME/bin/kafka-storage.sh" "$KAFKA_HOME/bin/kafka-server-start.sh" "$KAFKA_HOME/config/server.properties"
echo "Next (ONCE): bash realtime/kafka/init_kafka.sh"
