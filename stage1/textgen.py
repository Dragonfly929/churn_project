"""Templates for the clients' text messages (contact centre, chat, app reviews).

The true sentiment is known (simulation ground truth), so the Noise fundamental
can measure exactly how much typos, sarcasm, empty or contradictory messages
hurt the sentiment signal and how much denoising recovers.
"""
from __future__ import annotations

import numpy as np

POS = [
    "Mulțumesc pentru ajutor, totul a funcționat perfect",
    "Aplicația este foarte bună și rapidă",
    "Sunt mulțumit de serviciile băncii",
    "Consultantul a fost amabil și profesionist",
    "Transferul a ajuns repede, mulțumesc",
    "Recomand banca prietenilor, servicii excelente",
    "Problema a fost rezolvată rapid, mulțumesc frumos",
]
NEU = [
    "Aș dori să știu care este limita zilnică la card",
    "Cum pot schimba numărul de telefon în aplicație",
    "Vreau informații despre depozitele la termen",
    "Care este programul sucursalei de sâmbătă",
    "Am nevoie de un extras de cont pentru luna trecută",
    "Cum activez notificările pentru tranzacții",
    "Doresc să actualizez adresa de corespondență",
]
NEG = [
    "Comisioanele sunt prea mari, nu mai sunt mulțumit",
    "Aplicația nu funcționează de trei zile, este inacceptabil",
    "Am așteptat o oră la telefon și nimeni nu a răspuns",
    "Sunt nemulțumit, mi s-a blocat cardul fără motiv",
    "Dobânda la depozit este mult mai mică decât la alte bănci",
    "Vreau să închid contul, serviciile sunt slabe",
    "Am fost taxat de două ori pentru aceeași tranzacție, este o problemă gravă",
    "Personalul a fost nepoliticos, sunt dezamăgit",
]
# sarcasm: positive words, negative meaning (the hardest kind of noise)
SARCASM = [
    "Super, încă un comision nou, mulțumesc frumos",
    "Bravo, aplicația iar nu merge, excelent serviciu",
    "Minunat, am stat doar două ore la coadă",
    "Perfect, cardul blocat exact în concediu, mulțumesc",
]
# contradictory: mixed signals in one message
CONTRA = [
    "Aplicația e bună dar comisioanele sunt inacceptabile",
    "Mulțumesc pentru ajutor, dar probabil voi închide contul",
    "Servicii excelente, păcat că dobânda e atât de mică și sunt nemulțumit",
]
EN_NEG = ["Fees are too high, very disappointed", "App is not working, terrible service"]


def _typo(word: str, rng: np.random.Generator) -> str:
    if len(word) < 4:
        return word
    i = int(rng.integers(1, len(word) - 1))
    op = rng.integers(0, 3)
    if op == 0:   # swap
        return word[:i - 1] + word[i] + word[i - 1] + word[i + 1:]
    if op == 1:   # drop
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]  # double


STRIP = str.maketrans("ăâîșțşţĂÂÎȘȚ", "aaissstAAIST")


def make_message(sentiment: str, rng: np.random.Generator, noise_level: float = 1.0):
    """Return (text, noise_type). sentiment in {pos, neu, neg}."""
    r = rng.random()
    noise_type = "clean"
    if sentiment == "neg" and r < 0.10 * noise_level:
        text, noise_type = rng.choice(SARCASM), "sarcasm"
    elif sentiment == "neg" and r < 0.16 * noise_level:
        text, noise_type = rng.choice(CONTRA), "contradictory"
    elif sentiment == "neg" and r < 0.19 * noise_level:
        text, noise_type = rng.choice(EN_NEG), "english"
    else:
        text = rng.choice({"pos": POS, "neu": NEU, "neg": NEG}[sentiment])
    r2 = rng.random()
    if r2 < 0.04 * noise_level:
        return "", "empty"
    if r2 < 0.30 * noise_level:  # typos + missing diacritics + shouting
        words = [(_typo(w, rng) if rng.random() < 0.35 else w) for w in str(text).split()]
        text = " ".join(words)
        if rng.random() < 0.5:
            text = text.translate(STRIP)
        if rng.random() < 0.2:
            text = text.upper() + "!!!"
        noise_type = "typos" if noise_type == "clean" else noise_type + "+typos"
    return str(text), noise_type
