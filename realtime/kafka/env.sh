# shared settings for the Kafka scripts (source it)
KAFKA_VERSION=${KAFKA_VERSION:-4.3.1}
SCALA=2.13
KAFKA_HOME=${KAFKA_HOME:-$HOME/kafka/kafka_${SCALA}-${KAFKA_VERSION}}
KAFKA_DATA=${KAFKA_DATA:-$HOME/kafka/kraft-data}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP=${KAFKA_BOOTSTRAP:-localhost:9092}
