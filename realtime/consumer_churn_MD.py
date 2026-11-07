"""Terminal 3 of the golden rules: a plain consumer to WATCH the messages of a topic.

  python realtime/consumer_churn_MD.py --topic bank.events
  python realtime/consumer_churn_MD.py --topic bank.alerts --bus file
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realtime.bus import add_common_args, get_bus, run_info  # noqa: E402

ap = add_common_args(argparse.ArgumentParser())
ap.add_argument("--topic", default="bank.events")
ap.add_argument("--group", default=None)
a = ap.parse_args()
bus = get_bus(a.bus, a.bootstrap)
cons = bus.consumer(a.topic, a.group or f"watch-{run_info().get('run_id', 'manual')}")
print(f"watching {a.topic} on {bus.kind} ... (Ctrl+C to stop)")
while True:
    for m in cons.poll(500):
        print(json.dumps(m, ensure_ascii=False)[:300])
    cons.commit()
