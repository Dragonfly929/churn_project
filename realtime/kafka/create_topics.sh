#!/usr/bin/env bash
# Golden rule 2: create each topic ONCE; afterwards only check that it exists.
source "$(dirname "$0")/env.sh"
for T in bank.events bank.alerts bank.offers bank.outcomes; do
  if "$KAFKA_HOME/bin/kafka-topics.sh" --list --bootstrap-server "$BOOTSTRAP" | grep -qx "$T"; then
    echo "exists: $T"
  else
    "$KAFKA_HOME/bin/kafka-topics.sh" --create --topic "$T" --partitions 3 --replication-factor 1 --bootstrap-server "$BOOTSTRAP"
  fi
done
"$KAFKA_HOME/bin/kafka-topics.sh" --describe --topic bank.events --bootstrap-server "$BOOTSTRAP"
