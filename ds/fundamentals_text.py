"""The five parts of every fundamental, as text for the app and the report.

1. how it appears in this problem      -> APPEARS (fixed)
2. what was measured                   -> the tables returned by ds/fundamentals.py
3. what was observed                   -> observed(): sentences built from the measured numbers
4. mitigation „S-a efectuat …, deoarece …” -> "measures" returned with each result
5. answers to the lab questions        -> ANSWERS (fixed; refer to the measured results)

Order = the report template: 2.2.1 Bias, 2.2.2 Noise, 2.2.3 SMOTE, 2.2.4 Decision Boundaries,
2.2.5 Streaming, 2.2.6 RTAP.
"""
from __future__ import annotations

import pandas as pd

ORDER = ["bias", "noise", "smote", "boundaries", "streaming", "rtap"]
TITLES = {"bias": "Bias", "noise": "Noise", "smote": "SMOTE", "boundaries": "Decision boundaries",
          "streaming": "Streaming", "rtap": "RTAP"}

APPEARS = {
    "bias": "Clienții nu sunt tratați la fel pe grupuri de vârstă, gen și țară. Vârsta are și un proxy: "
            "ponderea operațiunilor digitale (digital_share_90d), puternic corelată cu vârsta. Un eșantion de "
            "antrenare în care un grup este subreprezentat (de exemplu 60+) produce un model care îl recunoaște mai slab.",
    "noise": "Feedback-ul textual are greșeli de scriere, sarcasm, mesaje contradictorii, în altă limbă sau goale; "
             "valorile numerice au erori de măsurare, iar eticheta poate fi greșită (data ieșirii înregistrată "
             "târziu). Comportamentul zilnic al clienților variază natural.",
    "smote": "Clienții care pleacă sunt clasa minoritară (aproximativ 16%). Fără echilibrare modelul învață mai "
             "ales clasa majoritară; SMOTE creează clienți sintetici prin interpolare între clienți reali care pleacă.",
    "boundaries": "Granița dintre „pleacă” și „rămâne” este o alegere de business: un client pierdut costă 1000 €, "
                  "o ofertă inutilă 60 €. Pragul 0,5 ignoră această asimetrie; modele diferite trasează granițe diferite.",
    "streaming": "Semnalele de plecare apar în timp (retrageri, reclamații, tăcere). Un scor lunar observă prea "
                 "târziu clienții al căror risc crește între două rulări; tăcerea nu trimite evenimente.",
    "rtap": "O alertă nu valorează nimic fără o acțiune. Ofertele costă, capacitatea echipei este limitată, iar "
            "rata de răspuns a fiecărei oferte pentru fiecare motiv de plecare nu este cunoscută dinainte.",
}

ANSWERS = {
    "bias": [
        ("Ce tip de bias ați observat în datele inițiale?",
         "Bias de reprezentare (grupul 60+ subreprezentat în varianta 2), bias prin atribute sensibile (vârstă, gen, "
         "țară) și bias prin proxy (digital_share_90d înlocuiește vârsta după eliminarea ei)."),
        ("Cum a afectat bias-ul performanța modelului?",
         "AUC-ul global se schimbă puțin, dar recall-ul diferă între grupurile de vârstă: un AUC bun poate ascunde "
         "un model nedrept. Diferența de recall este metrica relevantă, nu acuratețea."),
        ("Ce diferențe apar între modelul biasat și cel corectat?",
         "Modelul corectat (varianta 4: fără atribute sensibile și fără proxy) are diferența de recall pe vârstă "
         "mult mai mică, cu un AUC practic egal; recall-ul pierdut se recuperează prin alegerea pragului unic "
         "(iterația 3), nu prin praguri diferite pe grupuri."),
        ("Ce metode alternative de corectare ați putea explora?",
         "Reponderarea eșantioanelor, ThresholdOptimizer din fairlearn (varianta 6), constrângeri în antrenare "
         "(ExponentiatedGradient), colectarea de date suplimentare pentru grupul subreprezentat."),
        ("Ce diferențe apar în metricile de fairness? Selection rate și FPR?",
         "Selection rate arată cui i se trimit oferte; FPR arată cine este deranjat fără motiv. Un grup cu FPR mare "
         "primește mai multe contactări inutile, unul cu recall mic este ignorat când pleacă."),
        ("Ce alte caracteristici ar putea fi sensibile?",
         "Orice variabilă corelată cu vârsta, genul sau țara: canalul digital, tipul de card, produsele deținute, "
         "limba mesajelor."),
        ("Legătura cu corectitudinea și responsabilitatea în IA; integrarea în dezvoltare",
         "Fairness este un rând în matricea-contract (diferența maximă de recall ≤ 0,10), verificat la fiecare "
         "iterație ca orice altă cerință; decizia automată este doar o propunere (GDPR art. 22)."),
        ("Ce este mai rău, bias-ul sau noise-ul? Pot fi „bune”?",
         "Bias-ul este mai periculos: este sistematic și nu dispare cu mai multe date, pe când noise-ul se mai "
         "mediază. Noise-ul controlat poate fi util (augmentare, regularizare), iar un bias asumat (de exemplu "
         "prudența față de pierderea unui client) poate fi o alegere de business documentată."),
    ],
    "noise": [
        ("Ce model „înțelege” mai bine forma reală și de ce?",
         "Boosting-ul și pădurea aleatoare urmăresc relații neliniare (de exemplu tăcerea combinată cu scăderea "
         "soldului); regresia logistică vede doar direcții liniare, dar este mai stabilă la zgomot."),
        ("Cum reacționează fiecare model la noise? Unde se pierde semnalul și unde se păstrează?",
         "Zgomotul pe X scade AUC-ul treptat; zgomotul pe etichete (Y) scade F1 mult mai repede, pentru că modelul "
         "învață direct exemple greșite. Semnalul se păstrează în tendințe agregate (90 de zile), se pierde în "
         "valorile zilnice izolate și în textul brut."),
        ("Ce înseamnă „adevăr” într-un model care învață din zgomot? Ce se pierde, ce se păstrează?",
         "Adevărul este tiparul stabil care rămâne după mediere și curățare. Se pierd cazurile rare și nuanțele "
         "(sarcasmul fără denoising), se păstrează semnalele repetate la mulți clienți."),
    ],
    "smote": [
        ("SMOTE aplicat pe un subset de features păstrează performanța? De ce?",
         "Da, aproape la fel (tabelul subset): caracteristicile irelevante nu adaugă semnal, iar interpolarea "
         "pe mai puține dimensiuni creează clienți sintetici mai plauzibili."),
        ("Noise-ul afectează eficiența SMOTE? Cum?",
         "Da: cu 10% etichete greșite SMOTE multiplică exact exemplele greșite, iar precizia scade puternic."),
        ("În ce scenarii este SMOTE recomandat și în care nu?",
         "Recomandat când minoritatea este mică și cu caracteristici numerice continue; nu este recomandat cu etichete "
         "zgomotoase, cu variabile categoriale fără SMOTE-NC sau când ponderile claselor obțin același rezultat."),
        ("Cum validăm că modelul nu învață pe artefacte sintetice?",
         "SMOTE numai în fold-urile de antrenare, evaluare doar pe clienți reali, verificarea realismului "
         "(valori ne-întregi, distanța față de clienții reali); varianta greșită (SMOTE înainte de împărțire) "
         "umflă AUC-ul din validare."),
        ("Riscuri cu SMOTE pe toate features și cum alegem subsetul? Ce metrici sunt sensibile?",
         "Pe toate caracteristicile se interpolează și zgomot sau coduri fără sens; subsetul se alege după "
         "importanța SHAP și cunoașterea domeniului. Cele mai sensibile sunt precizia și F1, AUC-ul se schimbă puțin."),
        ("SMOTE înainte de noise; intensitatea noise-ului",
         "Zgomotul adăugat după SMOTE afectează și clienții sintetici, deci efectul se dublează; intensitatea se "
         "crește treptat (0,05 … 0,3) și se oprește unde F1 scade vizibil."),
        ("Augmentări necontrolate, optimizări obsesive; legătura cu under/overfitting",
         "Augmentarea necontrolată inventează clienți nerealiști (overfitting pe artefacte); optimizarea obsesivă "
         "a unei metrici pe aceleași date duce la overfitting pe setul de validare. Augmentarea poate fi și pe "
         "caracteristici (zgomot controlat, variante de text)."),
    ],
    "boundaries": [
        ("Ce diferențe observați între frontiere? Ce a învățat fiecare model?",
         "Regresia logistică trasează o graniță dreaptă (o direcție de risc), arborii trasează trepte, boosting-ul "
         "o graniță flexibilă, rețeaua neuronală o curbă netedă."),
        ("Cum ați trata incertitudinea dacă ați fi un algoritm?",
         "Clienții de lângă prag sunt incerți: îi trimit omului (decila 2–3) în loc să decid automat, iar pragul "
         "îl aleg după cost."),
        ("Ce observați când creșteți zgomotul? Cum se schimbă decizia?",
         "Granițele devin neregulate și instabile; modelele flexibile urmăresc zgomotul (overfitting), modelul "
         "liniar se schimbă puțin."),
        ("„A învăța prea puțin” sau „prea mult”? Prudent sau atent la tot?",
         "Curba de complexitate arată ambele: adâncime mică = underfitting, adâncime mare = AUC de antrenare mare și "
         "AUC de validare în scădere. Pentru retenție este preferabil modelul ușor prudent, cu prag ales din cost."),
        ("Ce am învățat? Ce întrebări am? Ce aleg ca model?",
         "Pragul contează mai mult decât diferențele mici dintre modele. Rămâne întrebarea cât de stabil este costul "
         "unei oferte. Aleg gradient boosting cu prag din matricea de cost."),
    ],
    "streaming": [
        ("De ce în timp real și nu lunar?",
         "Riscul crește între două rulări lunare; scorarea zilnică (evenimente + ceasul de închidere a zilei) prinde "
         "mai devreme clienții al căror risc apare după data de referință."),
        ("Cum tratăm tăcerea și alarmele false?",
         "Calea „ceas” rescorează zilnic și clienții fără evenimente; EWMA netezește scorul ca o singură zi "
         "zgomotoasă să nu genereze o ofertă."),
    ],
    "rtap": [
        ("De ce o politică adaptivă?",
         "Rata de succes a fiecărei oferte pe fiecare motiv nu este cunoscută; Thompson Sampling o învață din "
         "rezultate și echilibrează explorarea cu exploatarea."),
        ("Ce limite are acțiunea automată?",
         "Buget și capacitate zilnică, opt-out de marketing, oferta depinde doar de motiv (nu de gen, vârstă, țară), "
         "iar omul poate suprascrie orice decizie."),
        ("Ce face RAG? Unde caută, cine definește sursele, de ce e mai aproape de adevăr?",
         "Primește alerta și motivul, caută pasaje în baza de cunoștințe a băncii (playbook de retenție, produse, "
         "etică, metodologie), definită și întreținută de bancă, și formulează recomandarea doar din ele, cu "
         "sursele citate; nu inventează produse sau reguli."),
    ],
}


def _row(t: pd.DataFrame, col: str, start: str) -> pd.Series | None:
    m = t[t[col].astype(str).str.startswith(start)]
    return m.iloc[0] if len(m) else None


def observed(key: str, o: dict) -> list[str]:
    """Sentences (part 3) computed from the measured tables. Robust: skips what is missing."""
    t, out = o.get("table"), []
    try:
        if key == "bias":
            full, biased, mit, thr = (_row(t, "variant", s) for s in ("1.", "2.", "4.", "5."))
            if full is not None and mit is not None:
                out.append(f"Cu toate caracteristicile, diferența de recall între grupurile de vârstă este "
                           f"{full.gap_age_group:.3f}; după eliminarea atributelor sensibile și a proxy-ului scade la "
                           f"{mit.gap_age_group:.3f}, cu AUC {full.roc_auc:.3f} → {mit.roc_auc:.3f}.")
            if biased is not None:
                out.append(f"Eșantionul biasat (60+ subreprezentat) dă diferența {biased.gap_age_group:.3f} "
                           f"și recall {biased.recall:.3f}: AUC-ul ({biased.roc_auc:.3f}) nu arată problema.")
            if thr is not None and mit is not None:
                effect = ("deci reduc diferența" if thr.gap_age_group < mit.gap_age_group - 0.01
                          else "deci nu aduc un câștig de corectitudine: după eliminarea proxy-ului grupurile sunt "
                               "deja apropiate, iar un prag unic evită o decizie care depinde de vârstă")
                out.append(f"Pragurile pe grup de vârstă schimbă recall-ul {mit.recall:.3f} → {thr.recall:.3f} și "
                           f"diferența de vârstă {mit.gap_age_group:.3f} → {thr.gap_age_group:.3f}, {effect}.")
        elif key == "noise":
            x = t[t["noise"].str.startswith("X")]; y = t[t["noise"].str.startswith("Y")]
            if len(x) and len(y):
                out.append(f"Zgomot pe X (σ până la {x.level.max():g}): AUC {x.auc_mean.iloc[0]:.3f} → "
                           f"{x.auc_mean.iloc[-1]:.3f}, F1 {x.f1_mean.iloc[0]:.3f} → {x.f1_mean.iloc[-1]:.3f}.")
                out.append(f"Etichete greșite pe Y (până la {y.level.max():.0%}): AUC {y.auc_mean.iloc[0]:.3f} → "
                           f"{y.auc_mean.iloc[-1]:.3f}, F1 {y.f1_mean.iloc[0]:.3f} → {y.f1_mean.iloc[-1]:.3f}: zgomotul "
                           f"pe etichetă strică mai mult decât cel pe intrări.")
            tt = o.get("text_table")
            if tt is not None:
                tt = tt.reset_index() if "noise_type" not in tt.columns else tt
                a = _row(tt, "noise_type", "ALL")
                if a is not None:
                    out.append(f"Sentimentul din text: acuratețe {a.accuracy_raw:.3f} pe textul brut, "
                               f"{a.accuracy_denoised:.3f} după denoising; sarcasmul și mesajele în engleză erau "
                               f"citite greșit aproape complet.")
        elif key == "smote":
            wrong = _row(t, "variant", "SMOTE BEFORE")
            ok = t.dropna(subset=["roc_auc"])
            best = ok.loc[ok["expected_cost"].idxmin()]
            none = _row(t, "variant", "none")
            if none is not None:
                out.append(f"Fără echilibrare: recall {none.recall:.3f}, precizie {none.precision:.3f}; varianta cu "
                           f"costul cel mai mic este „{best.variant}” (recall {best.recall:.3f}, precizie "
                           f"{best.precision:.3f}, cost {best.expected_cost:,.0f} €).")
            if wrong is not None:
                out.append(f"SMOTE aplicat înainte de împărțire raportează un AUC de validare de {wrong.cv_auc:.3f} "
                           f"față de {best.cv_auc:.3f} aplicat corect: un rezultat fals, din scurgerea de date.")
            sub = o.get("subset_noise")
            if sub is not None and len(sub) >= 3:
                out.append(f"Cu 10% etichete greșite, SMOTE scade precizia la {sub.iloc[2].precision:.3f}: "
                           f"multiplică exemplele greșite.")
        elif key == "boundaries":
            best = t.loc[t["expected_cost"].idxmin()]
            out.append(f"Cel mai mic cost așteptat îl are {best.model} (AUC {best.roc_auc:.3f}, recall {best.recall:.3f} "
                       f"la pragul {best.threshold:.2f}); diferențele de AUC dintre modele sunt mici.")
            cm = o.get("confusion")
            if cm is not None and len(cm):
                lo, hi = cm.loc[cm["cost"].idxmin()], cm.iloc[-1]
                out.append(f"La pragul {hi.threshold:.2f} recall-ul este {hi.recall:.3f}, la pragul de cost "
                           f"{lo.threshold:.2f} este {lo.recall:.3f}, cu costul {hi.cost:,.0f} € → {lo.cost:,.0f} €.")
            dec = o.get("deciles")
            if dec is not None and len(dec):
                d = dec.reset_index() if "churn_rate" not in dec.columns else dec
                out.append(f"Decila de risc de sus are o rată de plecare de {d['churn_rate'].max():.0%}, cea de jos "
                           f"{d['churn_rate'].min():.0%}: scorul ordonează bine clienții.")
        elif key == "streaming":
            s, b = _row(t, "mode", "streaming daily + EWMA"), _row(t, "mode", "monthly batch")
            raw = _row(t, "mode", "streaming daily, no smoothing")
            sl = t[t["mode"].str.contains("only leavers") & t["mode"].str.startswith("streaming")]
            bl = t[t["mode"].str.contains("only leavers") & t["mode"].str.startswith("monthly")]
            if s is not None and b is not None:
                out.append(f"Pe toți cei care pleacă, streaming și batch sunt apropiate (recall înainte de ieșire "
                           f"{s.recall_before_exit:.3f} vs {b.recall_before_exit:.3f}).")
            if len(sl) and len(bl):
                out.append(f"Diferența apare la clienții al căror risc crește după data de referință: "
                           f"{sl.iloc[0].recall_before_exit:.3f} cu streaming vs {bl.iloc[0].recall_before_exit:.3f} "
                           f"cu batch lunar.")
            if s is not None and raw is not None:
                out.append(f"EWMA reduce alarmele false de la {int(raw['false_alarms (never leave)'])} la "
                           f"{int(s['false_alarms (never leave)'])} fără să piardă clienți prinși.")
        elif key == "rtap":
            ts, ex = _row(t, "policy", "Thompson"), _row(t, "policy", "expert")
            same = _row(t, "policy", "same offer")
            if ts is not None and ex is not None:
                out.append(f"Thompson Sampling: ROI {ts.ROI:.2f}, {int(ts.clients_saved)} clienți salvați; regulile "
                           f"expertului: ROI {ex.ROI:.2f}, {int(ex.clients_saved)} salvați.")
            if same is not None:
                out.append(f"Aceeași ofertă pentru toți: ROI {same.ROI:.2f}: potrivirea ofertei cu motivul plecării "
                           f"contează.")
            out.append("Politica adaptivă pornește fără nicio cunoaștere și învață din rezultate; regulile "
                       "expertului presupun că cineva cunoaște deja răspunsurile.")
    except Exception as e:  # noqa: BLE001
        out.append(f"(observation not available: {e})")
    return out
