"""RAG (Retrieval-Augmented Generation) for alerts.

1. Retrieve: the knowledge base (kb/*.md) is split into passages (one per '## ' section);
   the query = alert reason + top SHAP features; TF-IDF cosine similarity ranks passages.
2. Generate: the recommendation is written ONLY from the retrieved passages, with sources
   cited. Default = template (deterministic, offline). Optional: a local LLM through Ollama
   (set OLLAMA_MODEL, e.g. llama3.2) - it receives only the passages.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config

FEATURE_WORDS = {
    "days_since_last_tx": "silence days since last transaction", "days_since_last_msg": "silence no messages",
    "no_feedback_6m": "silence no messages 6 months", "nps_missing": "survey not answered silence",
    "silent_decline": "silent decline", "complaints_180d": "complaints dissatisfaction",
    "neg_msg_180d": "negative messages dissatisfaction", "avg_sentiment_180d": "negative sentiment",
    "nps_score": "low NPS score", "balance_change_3m": "balance change money withdrawal drained",
    "amount_out_90d": "large outgoing transfers", "net_flow_90d": "outgoing money", "amount_in_90d": "salary no longer received",
    "tx_trend": "falling transaction trend disengagement", "tx_count_90d": "fewer transactions",
    "is_active_member": "inactive member", "digital_share_90d": "digital sessions",
}
OFFER_OF_REASON = {"silence": "personal_call", "dissatisfaction": "fee_waiver",
                   "money_withdrawal": "deposit_bonus", "disengagement": "card_cashback", "profile": "personal_call"}


class KnowledgeBase:
    def __init__(self, folder=config.KB_DIR):
        self.passages = []
        for p in sorted(folder.glob("*.md")):
            text = p.read_text(encoding="utf-8")
            for sec in re.split(r"\n(?=## )", text):
                lines = sec.strip().splitlines()
                if not lines or lines[0].startswith("# "):
                    if len(lines) > 1 and lines[0].startswith("# "):
                        lines = lines[1:]
                    else:
                        continue
                title = lines[0].lstrip("# ").strip()
                body = " ".join(l.strip() for l in lines[1:] if l.strip())
                if body:
                    self.passages.append({"doc": p.name, "section": title, "text": body})
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.M = self.vec.fit_transform([f"{x['section']} {x['text']}" for x in self.passages])

    def retrieve(self, query: str, k: int = 3):
        sims = cosine_similarity(self.vec.transform([query]), self.M)[0]
        idx = np.argsort(-sims)[:k]
        return [{**self.passages[i], "score": float(sims[i])} for i in idx if sims[i] > 0]


_KB = None


def kb() -> KnowledgeBase:
    global _KB
    if _KB is None:
        _KB = KnowledgeBase()
    return _KB


def build_query(reason: str, top_features: list[str]) -> str:
    return f"{reason.replace('_', ' ')} " + " ".join(FEATURE_WORDS.get(f, f.replace('_', ' ')) for f in top_features)


def recommend(reason: str, top_features: list[str], risk: float, opt_out: int = 0, use_llm: bool | None = None) -> dict:
    q = build_query(reason, top_features)
    hits = kb().retrieve(q, k=3)
    eth = [p for p in kb().passages if p["doc"] == "ethics_policy.md" and
           ("opt-out" in p["text"] or "object" in p["section"].lower())]
    offer = OFFER_OF_REASON.get(reason, "personal_call")
    if opt_out:
        offer = "service_call_only"
        hits = hits + eth[:1]
    src = "; ".join(f"[{h['doc']} § {h['section']}]" for h in hits)
    text = (f"Risk {risk:.0%}, main reason: {reason}. "
            + (f"Recommendation: {hits[0]['text'][:260]}... " if hits else "")
            + ("The client opted out of marketing: no commercial offer. " if opt_out else "")
            + f"Proposed offer: {offer}. Sources: {src}")
    use_llm = use_llm if use_llm is not None else bool(os.environ.get("OLLAMA_MODEL"))
    if use_llm and hits:
        try:
            prompt = ("You are a bank retention assistant. Using ONLY the passages below, write a 2-sentence "
                      f"recommendation in Romanian for a client at risk {risk:.0%} with reason '{reason}'. "
                      "Cite passages as [doc § section].\n\n" +
                      "\n".join(f"[{h['doc']} § {h['section']}] {h['text']}" for h in hits))
            req = urllib.request.Request("http://localhost:11434/api/generate", method="POST",
                                         data=json.dumps({"model": os.environ.get("OLLAMA_MODEL", "llama3.2"),
                                                          "prompt": prompt, "stream": False}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                text = json.loads(r.read())["response"].strip() + f"\nSources: {src}"
        except Exception as e:  # noqa: BLE001
            text += f"  (LLM indisponibil: {e}; s-a folosit șablonul)"
    return {"offer": offer, "text": text, "sources": hits, "query": q}
