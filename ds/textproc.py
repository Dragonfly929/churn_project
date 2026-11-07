"""Text denoising + lexicon sentiment (Noise fundamental, text part).

raw_sentiment()      : naive lexicon on the raw text (exact word match).
                        Typos, missing diacritics, shouting and sarcasm break it.
denoise() + sentiment(): normalise -> fix typos against the vocabulary ->
                        handle negation, 'dar' (but) clauses and sarcasm cues.
"S-a efectuat denoising-ul textului, deoarece feedback-ul brut conține greșeli
 de tastare, lipsă de diacritice, sarcasm și mesaje goale, care inversează sau
 anulează semnalul de sentiment."
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from functools import lru_cache

POS_WORDS = {
    "multumesc", "multumit", "buna", "bun", "rapid", "rapida", "repede", "amabil", "profesionist",
    "excelent", "excelente", "perfect", "recomand", "rezolvata", "super", "bravo", "minunat",
    "frumos", "good", "great", "thanks",
}
NEG_WORDS = {
    "mari", "nemultumit", "inacceptabil", "inacceptabile", "nimeni", "blocat", "blocata",
    "mica", "mici", "inchid", "inchide", "slabe", "taxat", "problema", "grava", "nepoliticos",
    "dezamagit", "functioneaza", "merge", "comision", "comisionul", "comisioanele", "asteptat",
    "coada", "pacat", "terrible", "disappointed", "high", "fees", "not",
}
# words that turn a positive word into sarcasm when they appear in the same message
SARCASM_CUES = {"iar", "inca", "comision", "blocat", "coada", "ore", "doar", "concediu"}
NEGATIONS = {"nu", "n", "not", "nimeni", "niciodata"}
VOCAB = sorted(POS_WORDS | NEG_WORDS | SARCASM_CUES | NEGATIONS | {
    "aplicatia", "cardul", "contul", "dobanda", "depozit", "serviciile", "servicii", "banca",
    "telefon", "transferul", "consultantul", "personalul", "zile", "trei", "ora", "dar"})

# the raw lexicon has diacritics, like the dictionary someone would copy from the web
RAW_POS = {"mulțumesc", "mulțumit", "bună", "rapidă", "repede", "amabil", "profesionist",
           "excelente", "perfect", "recomand", "rezolvată", "super", "bravo", "minunat", "frumos"}
RAW_NEG = {"mari", "nemulțumit", "inacceptabil", "inacceptabile", "nimeni", "blocat", "mică",
           "închid", "slabe", "taxat", "problemă", "gravă", "nepoliticos", "dezamăgit",
           "comisioanele", "comision"}


def strip_accents(s: str) -> str:
    s = s.replace("ș", "s").replace("ş", "s").replace("ț", "t").replace("ţ", "t")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def is_garbled(text: str) -> bool:
    t = str(text or "")
    if not t.strip():
        return False
    letters = sum(ch.isalpha() for ch in t)
    return letters / max(1, len(t.replace(" ", ""))) < 0.6


@lru_cache(maxsize=50000)
def _fix(word: str) -> str:
    if word in VOCAB or len(word) < 4:
        return word
    m = difflib.get_close_matches(word, VOCAB, n=1, cutoff=0.8)
    return m[0] if m else word


def denoise(text: str) -> str:
    """Normalise case/diacritics/punctuation and correct typos."""
    t = str(text or "")
    if is_garbled(t):
        return ""
    t = strip_accents(t.lower())
    t = re.sub(r"[!?.]{2,}", " ", t)
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return " ".join(_fix(w) for w in t.split())


def _score_tokens(tokens: list[str]) -> float:
    s = 0.0
    for i, w in enumerate(tokens):
        val = 1.0 if w in POS_WORDS else (-1.0 if w in NEG_WORDS else 0.0)
        if val and i > 0 and tokens[i - 1] in NEGATIONS:
            val = -val if w not in {"functioneaza", "merge"} else -1.0
        elif w in {"functioneaza", "merge"}:
            val = -1.0 if (i > 0 and tokens[i - 1] in NEGATIONS) else 0.0
        s += val
    return s


def sentiment(clean_text: str) -> float:
    """Score in [-1, 1] on denoised text, with sarcasm and 'dar' (but) handling."""
    toks = clean_text.split()
    if not toks:
        return 0.0
    if "dar" in toks:  # the clause after "but" carries the meaning
        k = toks.index("dar")
        s = 0.3 * _score_tokens(toks[:k]) + 1.0 * _score_tokens(toks[k + 1:])
    else:
        s = _score_tokens(toks)
    pos = sum(w in POS_WORDS for w in toks)
    if pos and (set(toks) & SARCASM_CUES) and toks[0] in {"super", "bravo", "minunat", "perfect"}:
        s = -abs(s) - 1.0   # sarcasm: positive opener + complaint context
    return max(-1.0, min(1.0, s / 2.0))


def raw_sentiment(text: str) -> float:
    """What you get WITHOUT denoising: exact match on the raw words."""
    toks = re.findall(r"\w+", str(text or "").lower())
    s = sum(1 for w in toks if w in RAW_POS) - sum(1 for w in toks if w in RAW_NEG)
    return max(-1.0, min(1.0, s / 2.0))


def label_from_score(s: float, lo: float = -0.2, hi: float = 0.2) -> str:
    return "neg" if s <= lo else ("pos" if s >= hi else "neu")
