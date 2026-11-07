"""Message bus used by every real-time script. Two interchangeable backends:

  KafkaBus : Apache Kafka (KRaft) through kafka-python. key = client_id, JSON values,
             consumer groups, manual commit AFTER processing, resume from the committed offset.
  FileBus  : the no-Kafka FALLBACK (so the class demo cannot fail). Same semantics on disk:
             topic = folder realtime/bus/<topic>/, every flush = one JSONL file (append-only log),
             consumer group offsets = realtime/bus/_offsets/<group>__<topic>.json.
             Spark can read the same folders as a file stream.

Choose with --bus kafka|file (or env CHURN_BUS).
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

BUS_DIR = config.ROOT / "realtime" / "bus"
STATE = config.RT_STATE


def _json(o):
    return json.dumps(o, default=str, ensure_ascii=False).encode("utf-8")


# =============================================================== Kafka
class KafkaBus:
    kind = "kafka"

    def __init__(self, bootstrap=config.KAFKA_BOOTSTRAP):
        self.bootstrap = bootstrap
        self._producer = None

    def producer(self):
        if self._producer is None:
            from kafka import KafkaProducer
            self._producer = KafkaProducer(bootstrap_servers=self.bootstrap, key_serializer=lambda k: str(k).encode(),
                                           value_serializer=_json, linger_ms=5, acks=1, retries=5)
        return self._producer

    def send(self, topic, value: dict, key=None):
        self.producer().send(topic, key=key if key is not None else value.get("client_id", "0"), value=value)

    def flush(self):
        if self._producer:
            self._producer.flush()

    def consumer(self, topic, group):
        return _KafkaConsumer(self.bootstrap, topic, group)

    def ensure_topics(self, partitions=3):
        from kafka.admin import KafkaAdminClient, NewTopic
        adm = KafkaAdminClient(bootstrap_servers=self.bootstrap)
        have = set(adm.list_topics())
        new = [NewTopic(t, num_partitions=partitions, replication_factor=1) for t in config.TOPICS if t not in have]
        if new:
            adm.create_topics(new)
        adm.close()
        return [t.name for t in new]

    def stats(self, groups: dict) -> list[dict]:
        """Per topic: end offsets (messages), committed offsets and LAG of each consumer group."""
        from kafka import KafkaConsumer, TopicPartition
        from kafka.admin import KafkaAdminClient
        c = KafkaConsumer(bootstrap_servers=self.bootstrap)
        adm = KafkaAdminClient(bootstrap_servers=self.bootstrap)
        rows = []
        for topic in config.TOPICS:
            parts = c.partitions_for_topic(topic) or set()
            tps = [TopicPartition(topic, p) for p in parts]
            end = c.end_offsets(tps) if tps else {}
            total = sum(end.values())
            for g in groups.get(topic, []):
                try:
                    off = adm.list_consumer_group_offsets(g)
                    committed = sum(v.offset for tp, v in off.items() if tp.topic == topic and v.offset >= 0)
                except Exception:  # noqa: BLE001
                    committed = 0
                rows.append({"topic": topic, "partitions": len(parts), "messages": total, "group": g,
                             "committed": committed, "lag": total - committed})
            if not groups.get(topic):
                rows.append({"topic": topic, "partitions": len(parts), "messages": total, "group": "-",
                             "committed": None, "lag": None})
        c.close(); adm.close()
        return rows


class _KafkaConsumer:
    def __init__(self, bootstrap, topic, group):
        from kafka import KafkaConsumer
        self.c = KafkaConsumer(topic, bootstrap_servers=bootstrap, group_id=group, enable_auto_commit=False,
                               auto_offset_reset="latest", value_deserializer=lambda b: json.loads(b.decode("utf-8")),
                               max_poll_records=2000)

    def poll(self, timeout_ms=300) -> list[dict]:
        out = []
        for batch in self.c.poll(timeout_ms=timeout_ms).values():
            out += [m.value for m in batch]
        return out

    def commit(self):
        self.c.commit()  # commit AFTER processing (at-least-once)


# =============================================================== File fallback
class FileBus:
    kind = "file"

    def __init__(self, base=BUS_DIR):
        self.base = Path(base)
        self.pid = f"{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.buf: dict[str, list] = {}
        for t in config.TOPICS:
            (self.base / t).mkdir(parents=True, exist_ok=True)
        (self.base / "_offsets").mkdir(parents=True, exist_ok=True)

    def send(self, topic, value: dict, key=None):
        self.buf.setdefault(topic, []).append(value)
        if len(self.buf[topic]) >= 500:
            self._flush_topic(topic)

    def _flush_topic(self, topic):
        rows = self.buf.get(topic)
        if not rows:
            return
        d = self.base / topic
        name = f"{time.time_ns()}-{self.pid}.jsonl"
        tmp = d / f".tmp-{name}"
        tmp.write_bytes(b"\n".join(_json(r) for r in rows) + b"\n")
        tmp.rename(d / name)  # atomic: readers never see half a file
        self.buf[topic] = []

    def flush(self):
        for t in list(self.buf):
            self._flush_topic(t)

    def consumer(self, topic, group):
        return _FileConsumer(self.base, topic, group)

    def ensure_topics(self, partitions=1):
        return []

    def stats(self, groups: dict) -> list[dict]:
        rows = []
        for topic in config.TOPICS:
            files = sorted(p.name for p in (self.base / topic).glob("*.jsonl"))
            n = 0
            for f in files:
                with open(self.base / topic / f, "rb") as fh:
                    n += sum(1 for _ in fh)
            for g in groups.get(topic, []) or ["-"]:
                off = self.base / "_offsets" / f"{g}__{topic}.json"
                done = set(json.loads(off.read_text(encoding="utf-8"))) if off.exists() else set()
                rows.append({"topic": topic, "partitions": 1, "messages": n, "group": g,
                             "committed": f"{len(done)} files", "lag": f"{len(files) - len(done & set(files))} files"})
        return rows


class _FileConsumer:
    def __init__(self, base, topic, group):
        self.dir = base / topic
        self.off = base / "_offsets" / f"{group}__{topic}.json"
        self.done = set(json.loads(self.off.read_text(encoding="utf-8"))) if self.off.exists() else None
        if self.done is None:  # new group = start from "latest" (like Kafka's auto_offset_reset=latest)
            self.done = {p.name for p in self.dir.glob("*.jsonl")}
            self._save()
        self.pending: list[str] = []

    def _save(self):
        tmp = self.off.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(self.done)), encoding="utf-8")
        tmp.replace(self.off)

    def poll(self, timeout_ms=300) -> list[dict]:
        new = sorted(p.name for p in self.dir.glob("*.jsonl") if p.name not in self.done)
        if not new:
            time.sleep(timeout_ms / 1000)
            return []
        new = new[:20]
        out = []
        for f in new:
            out += [json.loads(l) for l in (self.dir / f).read_text(encoding="utf-8").splitlines() if l.strip()]
        self.pending = new
        return out

    def commit(self):
        self.done.update(self.pending)
        self.pending = []
        self._save()


def get_bus(kind: str | None = None, bootstrap: str | None = None):
    kind = (kind or os.environ.get("CHURN_BUS", "kafka")).lower()
    if kind == "file":
        return FileBus()
    return KafkaBus(bootstrap or config.KAFKA_BOOTSTRAP)


# =============================================================== shared run info (bank clock)
def run_info() -> dict:
    p = STATE / "run.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def new_run(seconds_per_day=config.SECONDS_PER_BANK_DAY, delay=10.0, days=config.STREAM_DAYS) -> dict:
    info = {"run_id": time.strftime("%Y%m%d-%H%M%S"), "t0": time.time() + delay, "seconds_per_day": seconds_per_day,
            "start_day": config.STREAM_START, "days": days}
    (STATE / "run.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def write_state(name: str, obj):
    p = STATE / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, default=str, indent=1), encoding="utf-8")
    tmp.replace(p)


def read_state(name: str, default=None):
    p = STATE / name
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def add_common_args(ap):
    ap.add_argument("--bus", default=os.environ.get("CHURN_BUS", "kafka"), choices=["kafka", "file"])
    ap.add_argument("--bootstrap", default=config.KAFKA_BOOTSTRAP, help="Kafka broker, e.g. 192.168.1.20:9092")
    return ap
