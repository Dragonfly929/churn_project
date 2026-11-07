#!/usr/bin/env bash
# Terminal 1: start the broker (every session). Keep this terminal open.
#   bash realtime/kafka/start_kafka.sh          # only this laptop
#   bash realtime/kafka/start_kafka.sh --lan    # other laptops on the same Wi-Fi connect to <this IP>:9092
source "$(dirname "$0")/env.sh"
CFG="$KAFKA_HOME/config/server.properties"
RUN_CFG="$HERE/server.run.properties"
cp "$CFG" "$RUN_CFG"
if [ "$1" = "--lan" ]; then
  IP=$(hostname -I | awk '{print $1}')
  sed -i "s#^listeners=.*#listeners=PLAINTEXT://0.0.0.0:9092,CONTROLLER://:9093#" "$RUN_CFG"
  sed -i "s#^advertised.listeners=.*#advertised.listeners=PLAINTEXT://${IP}:9092,CONTROLLER://${IP}:9093#" "$RUN_CFG"
  echo ">>> other laptops: ping $IP  then use --bootstrap $IP:9092   (open the port: sudo ufw allow 9092/tcp)"
fi
exec "$KAFKA_HOME/bin/kafka-server-start.sh" "$RUN_CFG"
