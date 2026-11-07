#!/usr/bin/env bash
# Topics, partitions, consumer groups, offsets and LAG (what to explain at the defense).
source "$(dirname "$0")/env.sh"
"$KAFKA_HOME/bin/kafka-topics.sh" --list --bootstrap-server "$BOOTSTRAP"
"$KAFKA_HOME/bin/kafka-topics.sh" --describe --bootstrap-server "$BOOTSTRAP" | grep -v "^$"
"$KAFKA_HOME/bin/kafka-consumer-groups.sh" --bootstrap-server "$BOOTSTRAP" --list
"$KAFKA_HOME/bin/kafka-consumer-groups.sh" --bootstrap-server "$BOOTSTRAP" --describe --all-groups 2>/dev/null
