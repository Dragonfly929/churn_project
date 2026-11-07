
import json
import os
import threading
from collections import deque
from datetime import datetime

import psutil
import streamlit as st
from kafka import KafkaConsumer, KafkaProducer

BROKER = "localhost:9092"
TOPIC = "system-metrics"


class KafkaLab:
    def __init__(self):
        self.lock = threading.Lock()
        self.producer_stop = threading.Event()
        self.consumer_stop = threading.Event()
        self.producer_thread = None
        self.consumer_thread = None
        self.latest = None
        self.history = deque(maxlen=100)
        self.sent = 0
        self.received = 0
        self.producer_error = None
        self.consumer_error = None

    def start_producer(self):
        if self.producer_thread and self.producer_thread.is_alive():
            return
        self.producer_stop.clear()
        self.producer_error = None
        self.producer_thread = threading.Thread(
            target=self.produce, daemon=True
        )
        self.producer_thread.start()

    def stop_producer(self):
        self.producer_stop.set()
        if self.producer_thread:
            self.producer_thread.join(timeout=3)

    def produce(self):
        producer = None
        try:
            producer = KafkaProducer(
                bootstrap_servers=BROKER,
                value_serializer=lambda x: json.dumps(x).encode("utf-8")
            )
            while not self.producer_stop.is_set():
                metrics = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "cpu_percent": psutil.cpu_percent(interval=0.5),
                    "memory": psutil.virtual_memory()._asdict(),
                    "disk": psutil.disk_usage(
                        os.path.abspath(os.sep)
                    )._asdict()
                }
                producer.send(TOPIC, metrics)
                producer.flush(timeout=5)
                with self.lock:
                    self.sent += 1
                    self.latest = metrics
                if self.producer_stop.wait(1.5):
                    break
        except Exception as e:
            self.producer_error = str(e)
        finally:
            if producer:
                producer.close()
            
    def start_consumer(self):
        if self.consumer_thread and self.consumer_thread.is_alive():
            return
        self.consumer_stop.clear()
        self.consumer_error = None
        self.consumer_thread = threading.Thread(
            target=self.consume, daemon=True
        )
        self.consumer_thread.start()

    def stop_consumer(self):
        self.consumer_stop.set()
        if self.consumer_thread:
            self.consumer_thread.join(timeout=3)

    def consume(self):
        consumer = None
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=BROKER,
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
                group_id="streamlit-kafka-lab"
            )
            while not self.consumer_stop.is_set():
                records = consumer.poll(timeout_ms=500)
                for batch in records.values():
                    for msg in batch:
                        data = msg.value
                        with self.lock:
                            self.received += 1
                            self.history.append({
                                "Seq": self.received,
                                "Ora": data.get("timestamp", ""),
                                "CPU (%)": data.get("cpu_percent", 0),
                                "RAM (%)": data.get("memory", {}).get("percent", 0),
                                "Disc (%)": data.get("disk", {}).get("percent", 0)
                            })
        except Exception as e:
            self.consumer_error = str(e)
        finally:
            if consumer:
                consumer.close()

    def snapshot(self):
        with self.lock:
            return (
                self.sent, self.received, self.latest,
                list(self.history),
                self.producer_error, self.consumer_error
            )


@st.cache_resource
def get_lab():
    return KafkaLab()


st.set_page_config(page_title="Kafka Lab", layout="wide")
lab = get_lab()


@st.fragment(run_every=2)
def show_lab():
    st.title("Apache Kafka — laborator de monitorizare")
    st.caption("Broker: localhost:9092 | Topic: system-metrics")

    tab_p, tab_c, tab_r, tab_ml = st.tabs([
        "1. Producer", "2. Consumer", "3. RTAP — Alertare în timp real",
        "4. RIVER — Online ML"
    ])

    with tab_p:
        st.subheader("Colectarea și transmiterea indicatorilor")
        st.write("CPU, memorie RAM și utilizarea discului sunt trimise periodic.")
        a, b = st.columns(2)
        if a.button("Pornește Producer", type="primary"):
            lab.start_producer()
        if b.button("Oprește Producer"):
            lab.stop_producer()

    with tab_c:
        st.subheader("Recepția și vizualizarea mesajelor")
        a, b = st.columns(2)
        if a.button("Pornește Consumer", type="primary"):
            lab.start_consumer()
        if b.button("Oprește Consumer"):
            lab.stop_consumer()

    sent, received, latest, history, p_err, c_err = lab.snapshot()
    m1, m2 = st.columns(2)
    m1.metric("Mesaje trimise", sent)
    m2.metric("Mesaje recepționate", received)

    if p_err:
        st.error("Eroare Producer: " + p_err)
    if c_err:
        st.error("Eroare Consumer: " + c_err)

    if latest:
        st.subheader("Ultimul mesaj generat")
        st.json(latest)

    if history:
        st.subheader("Istoricul recepției")
        display_history = [
            {k: row[k] for k in ("Ora", "CPU (%)", "RAM (%)", "Disc (%)")}
            for row in history
        ]
        st.dataframe(display_history, use_container_width=True, hide_index=True)
        st.line_chart(
            list(reversed(display_history)),
            x="Ora",
            y=["CPU (%)", "RAM (%)", "Disc (%)"]
        )

    with tab_r:
        st.subheader("RTAP — Real-Time Analytical Processing")
        st.write(
            "Tablou didactic pentru monitorizarea fluxului Kafka și declanșarea "
            "alertelor atunci când o caracteristică depășește pragul configurat."
        )
        st.info(
            "În această versiune, alerta este bazată pe praguri și persistența "
            "depășirii; nu reprezintă o predicție de defect realizată de un model ML. "
            "Valorile implicite sunt exemple și trebuie adaptate cerințelor de business."
        )

        col_cpu, col_ram, col_disk = st.columns(3)
        with col_cpu:
            cpu_limit = st.slider("Prag CPU (%)", 1, 100, 85, 1, key="rtap_cpu_limit")
        with col_ram:
            ram_limit = st.slider("Prag RAM (%)", 1, 100, 90, 1, key="rtap_ram_limit")
        with col_disk:
            disk_limit = st.slider("Prag disc (%)", 1, 100, 90, 1, key="rtap_disk_limit")
        persistence = st.slider(
            "Număr de eșantioane consecutive peste prag",
            min_value=1, max_value=10, value=3, step=1,
            key="rtap_persistence"
        )
        st.caption(
            "Exemplu: valoarea 3 declanșează alerta dacă ultimele 3 măsurători "
            "recepționate consecutiv sunt toate peste prag."
        )

        if not history:
            st.warning(
                "Nu există încă măsurători recepționate. Porniți Consumer-ul și Producer-ul "
                "din taburile lor; RTAP analizează mesajele consumate din topicul system-metrics."
            )
        else:
            metric_config = [
                ("CPU", "CPU (%)", cpu_limit),
                ("RAM", "RAM (%)", ram_limit),
                ("Disc", "Disc (%)", disk_limit),
            ]
            current = history[-1]
            status_cols = st.columns(3)
            active_now = set()
            alert_rows = []
            for idx, (name, field, limit) in enumerate(metric_config):
                value = float(current.get(field, 0) or 0)
                recent = history[-persistence:]
                persistent = len(recent) >= persistence and all(
                    float(row.get(field, 0) or 0) >= limit for row in recent
                )
                with status_cols[idx]:
                    st.metric(f"{name} curent", f"{value:.1f}%", f"Prag: {limit}%")
                    if persistent:
                        st.error("ALERTĂ — depășire persistentă")
                        active_now.add(name)
                        alert_rows.append({
                            "Ora": current.get("Ora", ""),
                            "Caracteristică": name,
                            "Valoare curentă (%)": round(value, 2),
                            "Prag (%)": limit,
                            "Eșantioane consecutive": persistence,
                            "Stare": "ALERTĂ"
                        })
                    elif value >= limit:
                        st.warning("Depășire momentană — se verifică persistența")
                    else:
                        st.success("Regim normal față de pragul configurat")

            if "rtap_active_metrics" not in st.session_state:
                st.session_state.rtap_active_metrics = set()
            if "rtap_alert_log" not in st.session_state:
                st.session_state.rtap_alert_log = []

            # Logăm o singură intrare la trecerea din normal în alertă.
            newly_active = active_now - st.session_state.rtap_active_metrics
            for name, field, limit in metric_config:
                if name in newly_active:
                    value = float(current.get(field, 0) or 0)
                    st.session_state.rtap_alert_log.append({
                        "Ora": current.get("Ora", ""),
                        "Caracteristică": name,
                        "Valoare (%)": round(value, 2),
                        "Prag (%)": limit,
                        "Motiv": f"{persistence} măsurători consecutive la/peste prag"
                    })
            st.session_state.rtap_active_metrics = active_now
            st.session_state.rtap_alert_log = st.session_state.rtap_alert_log[-50:]

            if alert_rows:
                st.subheader("Detalii despre alertele active")
                st.dataframe(alert_rows, use_container_width=True, hide_index=True)
                st.write(
                    "Interpretare: cel puțin una dintre caracteristici a depășit pragul "
                    "configurat în toate eșantioanele consecutive selectate. Într-un sistem "
                    "real, pragurile și durata persistenței trebuie calibrate pe date istorice "
                    "și pe costul alertelor false versus al incidentelor nedetectate."
                )
            else:
                st.success("Nu există alerte persistente active pentru pragurile selectate.")

            st.subheader("Istoricul alertelor RTAP")
            if st.session_state.rtap_alert_log:
                st.dataframe(
                    list(reversed(st.session_state.rtap_alert_log)),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.caption("Încă nu a fost înregistrată nicio tranziție către o alertă persistentă.")


    with tab_ml:
        st.subheader("River — învățare incrementală și detectarea anomaliilor")
        st.write(
            "Acest tab analizează aceleași măsurători recepționate de Consumer-ul Kafka. "
            "Detectorul River primește observațiile succesiv și își actualizează starea "
            "incremental, fără să fie reantrenat de la zero pe întregul istoric."
        )
        st.info(
            "River identifică observații neobișnuite, nu prezice automat defecte. "
            "Scorul de anomalie nu este o probabilitate de defect, iar pragul trebuie "
            "calibrat experimental pe date reprezentative."
        )

        try:
            from river import anomaly
        except ImportError:
            st.error("Biblioteca River nu este instalată în acest mediu Python.")
            st.code("python -m pip install river", language="bash")
            st.caption(
                "Rulați comanda în același mediu Python din care porniți Streamlit, "
                "apoi reporniți aplicația."
            )
        else:
            col_threshold, col_warmup = st.columns(2)
            with col_threshold:
                river_threshold = st.slider(
                    "Prag didactic pentru scorul de anomalie",
                    min_value=0.0, max_value=1.0, value=0.65, step=0.05,
                    key="river_score_threshold"
                )
            with col_warmup:
                river_warmup = st.slider(
                    "Număr minim de observații pentru încălzire",
                    min_value=5, max_value=50, value=20, step=5,
                    key="river_warmup"
                )

            st.caption(
                "Pragul 0,65 este doar un punct de pornire pentru demonstrație, nu o "
                "valoare universală. Calibrați-l pe fluxul real și analizați alertele false."
            )

            if "river_model" not in st.session_state:
                st.session_state.river_model = anomaly.HalfSpaceTrees(
                    n_trees=25, height=8, window_size=250, seed=42
                )
                st.session_state.river_last_seq = 0
                st.session_state.river_seen = 0
                st.session_state.river_results = []

            reset_col, info_col = st.columns([1, 3])
            with reset_col:
                reset_river = st.button("Resetează modelul River", key="river_reset")
            with info_col:
                st.caption(
                    "Resetarea șterge starea modelului și istoricul scorurilor River din "
                    "sesiunea curentă; Consumer-ul Kafka continuă separat."
                )

            if reset_river:
                st.session_state.river_model = anomaly.HalfSpaceTrees(
                    n_trees=25, height=8, window_size=250, seed=42
                )
                st.session_state.river_last_seq = 0
                st.session_state.river_seen = 0
                st.session_state.river_results = []

            # Procesăm fiecare mesaj nou o singură dată în această sesiune.
            new_rows = [
                row for row in history
                if int(row.get("Seq", 0)) > st.session_state.river_last_seq
            ]
            for row in new_rows:
                x = {
                    "cpu": max(0.0, min(1.0, float(row.get("CPU (%)", 0) or 0) / 100.0)),
                    "ram": max(0.0, min(1.0, float(row.get("RAM (%)", 0) or 0) / 100.0)),
                    "disk": max(0.0, min(1.0, float(row.get("Disc (%)", 0) or 0) / 100.0)),
                }
                # Evaluăm observația înainte de a actualiza detectorul.
                score = float(st.session_state.river_model.score_one(x))
                st.session_state.river_model.learn_one(x)
                st.session_state.river_seen += 1
                st.session_state.river_results.append({
                    "Seq": int(row.get("Seq", 0)),
                    "Ora": row.get("Ora", ""),
                    "CPU (%)": round(x["cpu"] * 100, 1),
                    "RAM (%)": round(x["ram"] * 100, 1),
                    "Disc (%)": round(x["disk"] * 100, 1),
                    "Scor anomalie": round(score, 4),
                    "Stare": (
                        "Încălzire"
                        if st.session_state.river_seen < river_warmup
                        else ("ANOMALIE posibilă" if score >= river_threshold else "Obișnuit")
                    )
                })
                st.session_state.river_last_seq = int(row.get("Seq", 0))
                st.session_state.river_results = st.session_state.river_results[-100:]

            results = st.session_state.river_results
            seen = st.session_state.river_seen
            if not history:
                st.warning(
                    "Nu există măsurători încă. Porniți Consumer-ul și Producer-ul, "
                    "apoi reveniți la acest tab."
                )
            elif not results:
                st.warning(
                    "Nu există observații noi de procesat. Dacă tocmai ați resetat modelul, "
                    "așteptați următorul mesaj Kafka."
                )
            else:
                last_result = results[-1]
                anomaly_now = (
                    seen >= river_warmup
                    and last_result["Scor anomalie"] >= river_threshold
                )
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Observații procesate", seen)
                k2.metric("Scor curent", f"{last_result['Scor anomalie']:.4f}")
                k3.metric("CPU curent", f"{last_result['CPU (%)']:.1f}%")
                k4.metric("RAM curent", f"{last_result['RAM (%)']:.1f}%")

                if seen < river_warmup:
                    st.warning(
                        f"Perioadă de încălzire: {seen}/{river_warmup} observații. "
                        "Nu interpretăm încă scorul ca alertă stabilă."
                    )
                elif anomaly_now:
                    st.warning(
                        "Scorul curent depășește pragul didactic: observația este "
                        "neobișnuită față de tiparele învățate de detector. Verificați contextul."
                    )
                else:
                    st.success(
                        "Scorul curent nu depășește pragul configurat. Acest lucru nu "
                        "garantează că sistemul este lipsit de defecte."
                    )

                st.subheader("Istoricul scorurilor River")
                st.dataframe(
                    list(reversed(results)), use_container_width=True, hide_index=True
                )
                st.line_chart(
                    list(reversed(results)), x="Ora", y=["Scor anomalie"]
                )

                st.subheader("Cum interpretăm rezultatul?")
                st.markdown(
                    "1. **Scorare:** detectorul evaluează cât de neobișnuită pare observația.  "
                    "\n2. **Învățare incrementală:** observația este apoi introdusă în detector.  "
                    "\n3. **Prag:** pragul transformă scorul într-un semnal didactic de alertare; "
                    "el trebuie calibrat, nu presupus corect.  "
                    "\n4. **Verificare:** pentru a demonstra detectarea defectelor, avem nevoie de "
                    "etichete sau confirmări reale și de evaluarea alertelor false și ratate."
                )
                st.caption(
                    "Notă: modelul River și cursorul său sunt păstrate în sesiunea Streamlit. "
                    "Istoricul vizibil este limitat la ultimele 100 de rezultate. Dacă aplicația "
                    "se repornește sau sesiunea se resetează, modelul didactic pornește din nou."
                )


show_lab()
