"""
Analyse des avis Google Maps des agences SG, à partir du JSONL produit par le scraper.

  (1) Sentiment        : polarité (positif / neutre / négatif) + intensité (0 à 1)
  (2) Thèmes négatifs  : topic modeling (NMF sur TF-IDF) + tagging par dictionnaire de thèmes
                         -> irritants opérationnels (conseillers, attente, produits, frais...)
  (3) Biais            : volume par agence, couverture, distribution des notes (effet extrêmes),
                         avis avec/sans texte, réponses du propriétaire, ancienneté, tests

Utilisation :
    python analyse_avis_sg.py                      # lit avis_agences.jsonl
    python analyse_avis_sg.py autre_fichier.jsonl

Dépendances : pip install pandas numpy scikit-learn matplotlib scipy
Recommandé pour le sentiment : pip install transformers torch
  -> sans transformers, le script bascule sur un lexique français simplifié (plus grossier).

Sorties dans le dossier analyse_avis/ : CSV (séparateur ';') + graphiques PNG.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # génère des PNG sans fenêtre
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from scipy import stats
except ImportError:  # les tests statistiques sont optionnels
    stats = None

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
INPUT_JSONL = sys.argv[1] if len(sys.argv) > 1 else "avis_agences.jsonl"
OUT_DIR = Path("analyse_avis")

SENTIMENT_BACKEND = "auto"   # "auto" | "transformers" | "lexique"
HF_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"  # notes 1-5 étoiles, multilingue
POS_THRESHOLD = 0.25         # score >= 0.25  -> positif
NEG_THRESHOLD = -0.25        # score <= -0.25 -> négatif

# Définition d'un avis "négatif" pour l'analyse thématique :
# note <= 2 étoiles OU sentiment du texte <= NEG_THRESHOLD
NEG_MAX_RATING = 2

N_TOPICS = 6                 # nombre de topics NMF (réduit automatiquement si peu d'avis)
MIN_REVIEWS_PER_AGENCY = 30  # en dessous : agence signalée "faible volume"

# --------------------------------------------------------------------------- #
# Dictionnaire de thèmes (irritants opérationnels) : à adapter à ton vocabulaire
# Les mots-clés sont des débuts de mot (ex. "attend" capte attendre, attendu, attente...)
# --------------------------------------------------------------------------- #
THEMES = {
    "Compétence des conseillers": [
        "compétent", "incompétent", "incompétence", "conseiller", "conseillère", "conseil",
        "erreur", "ignor", "connaissance", "renseign", "formation", "n'y connaît",
        "ne sait pas", "ne savent pas", "mal informé", "mauvaise information",
    ],
    "Attitude / accueil": [
        "accueil", "désagréable", "impoli", "méprisant", "mépris", "arrogan", "irrespect",
        "agressi", "froid", "hautain", "malpoli", "aimable", "sourire", "écoute", "respect",
    ],
    "Temps d'attente / disponibilité": [
        "attend", "attente", "file", "queue", "rendez-vous", "rdv", "délai", "longtemps",
        "heures", "minutes", "injoignable", "joindre", "répond", "répondeur", "rappel",
        "téléphone", "disponib", "lenteur", "lent",
    ],
    "Produits & services": [
        "prêt", "crédit", "assurance", "carte", "livret", "épargne", "placement", "immobilier",
        "produit", "offre", "contrat", "souscri", "mutuelle", "hypothèque", "investissement",
    ],
    "Frais & tarification": [
        "frais", "tarif", "commission", "cher", "prélèvement", "facturation", "facturé",
        "agios", "coût", "abonnement", "pénalité", "taux",
    ],
    "Opérations & gestion de compte": [
        "virement", "chèque", "retrait", "dépôt", "plafond", "bloqu", "blocage", "clôtur",
        "découvert", "rejet", "opération", "compte", "ouverture de compte", "fermeture du compte",
    ],
    "Digital / appli / distributeur": [
        "application", "appli", "site", "internet", "en ligne", "distributeur", "dab",
        "automate", "borne", "mot de passe", "connexion", "digital", "mobile",
    ],
    "Organisation de l'agence": [
        "horaire", "fermé", "fermeture", "ouverture", "turnover", "changent", "effectif",
        "manque de personnel", "sans rendez-vous", "accessible", "parking", "locaux",
    ],
    "Réclamations & suivi de dossier": [
        "réclamation", "plainte", "médiateur", "service client", "courrier", "suivi", "dossier",
        "promesse", "litige", "fraude", "arnaque", "escroquerie", "sans réponse", "aucune réponse",
    ],
}
THEME_RX = {
    name: re.compile(r"\b(?:" + "|".join(re.escape(k) for k in kws) + ")", re.I)
    for name, kws in THEMES.items()
}

FRENCH_STOPWORDS = set("""
a à au aux avec ce ces cet cette dans de des du elle elles en et est été être eu il ils je la le les
leur leurs lui ma mais me même mes moi mon ne nos notre nous on ou par pas pour que qui sa se ses si
son sur ta te tes toi ton tu un une vos votre vous ai as avons avez ont avais avait avions avaient suis
es sommes êtes sont était étais étaient fait faire faits fais fait plus très tout tous toute toutes
comme aussi bien donc alors puis après avant depuis encore déjà jamais rien ça cela celui ceux dont où
quand comment pourquoi car entre chez sans sous vers peu trop bon bonne ete etc cest jai nest
quil quelle quelque quelques lors ainsi autre autres chaque deux trois fois ans mois jour jours
banque agence agences sg société générale societe generale bancaire client clients
""".split())

# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
def to_float(x):
    if x is None or x == "":
        return np.nan
    try:
        return float(str(x).replace(",", ".").strip())
    except ValueError:
        return np.nan


def parse_age_months(s):
    """'il y a 5 mois' -> 5 ; 'il y a un an' -> 12 ; 'il y a 2 semaines' -> ~0.5."""
    if not isinstance(s, str):
        return np.nan
    s = s.lower().replace("\xa0", " ")
    m = re.search(r"il y a (un|une|\d+)\s*(jour|semaine|mois|an)", s)
    if not m:
        return np.nan
    n = 1 if m.group(1) in ("un", "une") else int(m.group(1))
    return n * {"jour": 1 / 30, "semaine": 7 / 30, "mois": 1, "an": 12}[m.group(2)]


def load_data(path):
    agencies = {}
    reviews = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a = json.loads(line)
            p = a.get("place") or {}
            aid = str(a["id"])
            # si l'agence apparaît plusieurs fois (relance), on garde la dernière ligne
            agencies[aid] = {
                "agence_id": aid,
                "adresse": a.get("adresse_input"),
                "place_name": p.get("name"),
                "rating_fiche": to_float(p.get("rating")),
                "total_fiche": to_float(p.get("total_reviews")),
                "status": a.get("status"),
            }
            reviews[aid] = [
                {
                    "agence_id": aid,
                    "review_id": r.get("review_id"),
                    "rating": to_float(r.get("rating")),
                    "date": r.get("date"),
                    "text": (r.get("text") or "").strip(),
                    "owner_response": (r.get("owner_response") or "").strip(),
                }
                for r in (a.get("reviews") or [])
            ]
    ag = pd.DataFrame(agencies.values())
    df = pd.DataFrame([r for rs in reviews.values() for r in rs])
    if df.empty:
        sys.exit("Aucun avis dans le fichier d'entrée.")

    ag["n_scraped"] = ag["agence_id"].map(df.groupby("agence_id").size()).fillna(0).astype(int)
    ag["label"] = ag["agence_id"] + " - " + ag["adresse"].fillna("").str.slice(0, 28)
    df = df.merge(ag[["agence_id", "label"]], on="agence_id", how="left")

    df["has_text"] = df["text"].str.len() >= 3
    df["text_len"] = df["text"].str.len()
    df["has_response"] = df["owner_response"].str.len() > 0
    df["age_months"] = df["date"].map(parse_age_months)
    return df, ag


# --------------------------------------------------------------------------- #
# (1) Sentiment
# --------------------------------------------------------------------------- #
TOKEN_RE = re.compile(r"[a-zàâäçéèêëîïôöùûüÿœ]+")
NEGATIONS = {"pas", "jamais", "aucun", "aucune", "sans", "ni"}
INTENSIFIERS = {"très", "tres", "vraiment", "tellement", "extrêmement", "totalement", "complètement"}
ATTENUATORS = {"assez", "plutôt", "peu", "moyennement", "relativement"}

LEXIQUE = {
    # positifs
    "excellent": 3, "parfait": 3, "génial": 3, "formidable": 3, "impeccable": 3, "exceptionnel": 3,
    "super": 2, "bravo": 2, "satisf": 2, "ravi": 2, "recommand": 2, "professionnel": 2,
    "aimable": 2, "agréable": 2, "sympa": 2, "efficace": 2, "compétent": 2, "accueillant": 2,
    "chaleureux": 2, "souriant": 2, "serviable": 2, "top": 2, "merci": 1, "content": 1,
    "rapide": 1, "disponible": 1, "écoute": 1, "patient": 1, "confiance": 1, "qualité": 1,
    "bien": 1, "bon": 1, "bonne": 1, "sérieux": 1, "réactif": 2,
    # négatifs
    "nul": -3, "catastroph": -3, "horrible": -3, "scandal": -3, "inadmissible": -3,
    "incompétent": -3, "incompétence": -3, "arnaqu": -3, "honteux": -3, "déplorable": -3,
    "lamentable": -3, "inacceptable": -3, "escroc": -3, "méprisant": -3, "mépris": -3,
    "pire": -3, "abus": -3, "fuyez": -3, "mauvais": -2, "mauvaise": -2, "déç": -2,
    "décevant": -2, "désagréable": -2, "impoli": -2, "injoignable": -2, "impossible": -2,
    "problème": -2, "erreur": -2, "refus": -2, "bloqu": -2, "clôtur": -2, "médiocre": -2,
    "inefficace": -2, "arrogan": -2, "agressi": -2, "lent": -1, "lenteur": -1, "attente": -1,
    "attend": -1, "cher": -1, "frais": -1, "difficile": -1, "compliqué": -1, "dommage": -1,
    "panne": -2, "interminable": -2, "jamais": -1, "inutile": -2, "incapable": -3,
}


def lex_weight(tok):
    for stem, w in LEXIQUE.items():
        if len(stem) >= 5:
            if tok.startswith(stem):
                return w
        elif tok == stem or tok == stem + "s":
            return w
    return 0


def lexicon_score(text):
    toks = TOKEN_RE.findall(text.lower())
    total = 0.0
    for i, t in enumerate(toks):
        w = lex_weight(t)
        if w == 0:
            continue
        window = toks[max(0, i - 3):i]
        if any(x in NEGATIONS for x in window):
            w = -0.8 * w
        if any(x in INTENSIFIERS for x in window):
            w *= 1.4
        elif any(x in ATTENUATORS for x in window):
            w *= 0.6
        total += w
    if total != 0:
        total += np.sign(total) * 0.3 * min(text.count("!"), 3)
    return float(np.tanh(total / 3))


def transformer_scores(texts, batch_size=16):
    """Score dans [-1, 1] = (note attendue sur 5 - 3) / 2, à partir des probabilités du modèle."""
    from transformers import pipeline

    clf = pipeline("text-classification", model=HF_MODEL)
    out = clf(list(texts), batch_size=batch_size, top_k=None, truncation=True, max_length=512)
    scores = []
    for probs in out:
        expected = sum(int(d["label"][0]) * d["score"] for d in probs)  # "4 stars" -> 4
        scores.append((expected - 3) / 2)
    return scores


def add_sentiment(df):
    backend = SENTIMENT_BACKEND
    if backend == "auto":
        try:
            import transformers  # noqa: F401
            backend = "transformers"
        except ImportError:
            backend = "lexique"
    print(f"[sentiment] méthode : {backend}")

    df["sentiment_score"] = np.nan
    mask = df["has_text"]
    texts = df.loc[mask, "text"]
    if backend == "transformers":
        df.loc[mask, "sentiment_score"] = transformer_scores(texts)
    else:
        df.loc[mask, "sentiment_score"] = texts.map(lexicon_score)

    s = df["sentiment_score"]
    df["polarity"] = np.select(
        [s >= POS_THRESHOLD, s <= NEG_THRESHOLD, s.notna()],
        ["positif", "négatif", "neutre"],
        default="sans texte",
    )
    df["intensity"] = s.abs()
    df["intensity_level"] = pd.cut(
        df["intensity"], [-0.01, 0.33, 0.66, 1.01], labels=["faible", "modérée", "forte"]
    )
    # écart entre ce que dit le texte et la note donnée (ex : texte très négatif mais 5 étoiles)
    df["gap_text_vs_rating"] = df["sentiment_score"] - (df["rating"] - 3) / 2
    return df


def sentiment_outputs(df, ag):
    t = df[df["has_text"]]
    by_agency = (
        t.groupby("label")
        .agg(
            n_avis_texte=("text", "size"),
            score_moyen=("sentiment_score", "mean"),
            score_median=("sentiment_score", "median"),
            intensite_moyenne=("intensity", "mean"),
            note_moyenne=("rating", "mean"),
            pct_positif=("polarity", lambda x: (x == "positif").mean()),
            pct_neutre=("polarity", lambda x: (x == "neutre").mean()),
            pct_negatif=("polarity", lambda x: (x == "négatif").mean()),
            pct_intensite_forte=("intensity_level", lambda x: (x == "forte").mean()),
        )
        .round(3)
        .reset_index()
    )
    save_csv(by_agency, "1_sentiment_par_agence.csv")

    incoh = df[df["gap_text_vs_rating"].abs() >= 1.0].sort_values(
        "gap_text_vs_rating", key=lambda x: -x.abs()
    )
    save_csv(
        incoh[["label", "rating", "sentiment_score", "gap_text_vs_rating", "text"]].head(100),
        "1_avis_incoherents_texte_vs_note.csv",
    )

    cols = ["label", "review_id", "rating", "date", "text", "sentiment_score",
            "polarity", "intensity", "intensity_level"]
    save_csv(df[cols], "1_avis_scores.csv")

    # Graphiques
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(t["sentiment_score"], bins=30, color="#4a7fb5")
    ax[0].axvline(NEG_THRESHOLD, color="red", ls="--")
    ax[0].axvline(POS_THRESHOLD, color="green", ls="--")
    ax[0].set(title="Distribution du score de sentiment (-1 = très négatif, +1 = très positif)",
              xlabel="score", ylabel="nb d'avis")
    groups = [t.loc[t["rating"] == r, "sentiment_score"].dropna() for r in [1, 2, 3, 4, 5]]
    ax[1].boxplot(groups)
    ax[1].set_xticklabels(["1★", "2★", "3★", "4★", "5★"])
    ax[1].set(title="Score de sentiment du texte selon la note", ylabel="score")
    fig.tight_layout()
    save_fig(fig, "1_sentiment_distribution.png")

    pol = (
        t.groupby("label")["polarity"].value_counts(normalize=True).unstack().fillna(0)
        .reindex(columns=["négatif", "neutre", "positif"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(pol) + 2))
    pol.plot.barh(stacked=True, ax=ax, color=["#d9534f", "#bbbbbb", "#5cb85c"])
    ax.set(title="Polarité des avis par agence", xlabel="part des avis avec texte", ylabel="")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, "1_sentiment_par_agence.png")

    print("\n=== (1) SENTIMENT ===")
    print(t["polarity"].value_counts(normalize=True).round(3).to_string())
    print(f"Intensité moyenne : {t['intensity'].mean():.2f} | "
          f"avis d'intensité forte : {(t['intensity_level'] == 'forte').mean():.1%}")
    print(f"Avis où texte et note se contredisent fortement : {len(incoh)}")
    return by_agency


# --------------------------------------------------------------------------- #
# (2) Thèmes des avis négatifs
# --------------------------------------------------------------------------- #
def tag_themes(df_neg):
    for name, rx in THEME_RX.items():
        df_neg[name] = df_neg["text"].map(lambda s, rx=rx: bool(rx.search(s)))
    theme_cols = list(THEMES)
    df_neg["themes"] = df_neg[theme_cols].apply(
        lambda r: " | ".join(c for c in theme_cols if r[c]) or "Autre / non classé", axis=1
    )
    return df_neg


def run_nmf(df_neg):
    n = len(df_neg)
    k = int(min(N_TOPICS, max(2, n // 15)))
    vec = TfidfVectorizer(
        stop_words=list(FRENCH_STOPWORDS),
        ngram_range=(1, 2),
        min_df=2 if n >= 30 else 1,
        max_df=0.7,
        token_pattern=r"(?u)\b[^\W\d_]{3,}\b",
    )
    X = vec.fit_transform(df_neg["text"])
    if X.shape[1] < k:
        return None
    model = NMF(n_components=k, init="nndsvda", random_state=0, max_iter=600)
    W = model.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())

    rows = []
    for i, comp in enumerate(model.components_):
        top = terms[comp.argsort()[::-1][:12]]
        # libellé suggéré = thème du dictionnaire qui recoupe le plus les mots du topic
        overlap = {th: sum(bool(rx.search(t)) for t in top) for th, rx in THEME_RX.items()}
        best = max(overlap, key=overlap.get)
        rows.append({
            "topic": i,
            "libelle_suggere": best if overlap[best] > 0 else "À interpréter",
            "termes_principaux": ", ".join(top),
        })
    topics = pd.DataFrame(rows)
    dominant = W.argmax(axis=1)
    topics["n_avis"] = [(dominant == i).sum() for i in range(k)]
    topics["part"] = (topics["n_avis"] / n).round(3)
    df_neg["topic_nmf"] = dominant
    df_neg["topic_nmf_poids"] = W.max(axis=1)
    return topics


def topic_outputs(df):
    neg = df[
        df["has_text"] & ((df["rating"] <= NEG_MAX_RATING) | (df["sentiment_score"] <= NEG_THRESHOLD))
    ].copy()
    print(f"\n=== (2) THÈMES DES AVIS NÉGATIFS ({len(neg)} avis négatifs avec texte) ===")
    if len(neg) < 8:
        print("Trop peu d'avis négatifs pour une analyse thématique.")
        return
    neg = tag_themes(neg)

    # a) thèmes par dictionnaire (multi-étiquettes : un avis peut cumuler plusieurs thèmes)
    theme_cols = list(THEMES)
    summary = pd.DataFrame({
        "theme": theme_cols,
        "n_avis": [int(neg[c].sum()) for c in theme_cols],
        "part_des_avis_negatifs": [neg[c].mean() for c in theme_cols],
        "note_moyenne": [neg.loc[neg[c], "rating"].mean() for c in theme_cols],
        "intensite_moyenne": [neg.loc[neg[c], "intensity"].mean() for c in theme_cols],
    }).sort_values("n_avis", ascending=False).round(3)
    n_other = int((neg["themes"] == "Autre / non classé").sum())
    summary.loc[len(summary)] = ["Autre / non classé", n_other, round(n_other / len(neg), 3), np.nan, np.nan]
    save_csv(summary, "2_themes_irritants.csv")
    print(summary.to_string(index=False))

    by_agency = (
        neg.groupby("label")[theme_cols].mean().round(3)
        .assign(n_avis_negatifs=neg.groupby("label").size())
        .reset_index()
    )
    save_csv(by_agency, "2_themes_par_agence.csv")

    # b) verbatims représentatifs (les plus intenses) par thème
    verb = []
    for c in theme_cols:
        sub = neg[neg[c]].sort_values("intensity", ascending=False).head(5)
        for _, r in sub.iterrows():
            verb.append({"theme": c, "agence": r["label"], "note": r["rating"],
                         "intensite": round(r["intensity"], 2), "avis": r["text"]})
    save_csv(pd.DataFrame(verb), "2_verbatims_par_theme.csv")

    # c) topics non supervisés (NMF) pour repérer ce que le dictionnaire ne capte pas
    topics = run_nmf(neg)
    if topics is not None:
        save_csv(topics, "2_topics_nmf.csv")
        print("\nTopics découverts (NMF) :")
        print(topics.to_string(index=False))
        save_csv(
            neg[["label", "rating", "text", "themes", "topic_nmf", "topic_nmf_poids"]],
            "2_avis_negatifs_etiquetes.csv",
        )
    else:
        save_csv(neg[["label", "rating", "text", "themes"]], "2_avis_negatifs_etiquetes.csv")

    # Graphiques
    s = summary.sort_values("n_avis")
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(s) + 2))
    ax.barh(s["theme"], s["part_des_avis_negatifs"], color="#d9534f")
    ax.set(title="Irritants cités dans les avis négatifs (un avis peut cumuler plusieurs thèmes)",
           xlabel="part des avis négatifs")
    fig.tight_layout()
    save_fig(fig, "2_themes_irritants.png")

    heat = by_agency.set_index("label")[theme_cols]
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(heat) + 3))
    im = ax.imshow(heat.values, aspect="auto", cmap="Reds")
    ax.set_xticks(range(len(theme_cols)))
    ax.set_xticklabels(theme_cols, rotation=40, ha="right")
    ax.set_yticks(range(len(heat)))
    n_neg = by_agency.set_index("label")["n_avis_negatifs"]
    ax.set_yticklabels([f"{lab} (n={n_neg[lab]})" for lab in heat.index])  # n affiché : petits effectifs = % instables
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.values[i, j]:.0%}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="part des avis négatifs de l'agence")
    ax.set_title("Irritants par agence")
    fig.tight_layout()
    save_fig(fig, "2_themes_par_agence.png")


# --------------------------------------------------------------------------- #
# (3) Biais
# --------------------------------------------------------------------------- #
def bias_outputs(df, ag):
    d = df.dropna(subset=["rating"]).copy()
    d["rating"] = d["rating"].astype(int)
    stars = [1, 2, 3, 4, 5]
    print("\n=== (3) BIAIS ===")

    # a) distribution globale
    dist = d["rating"].value_counts(normalize=True).reindex(stars).fillna(0)
    extremes = dist[1] + dist[5]
    print("Distribution globale des notes :", (dist * 100).round(1).to_dict())
    print(f"Part d'avis extrêmes (1★+5★) : {extremes:.1%} | ratio 5★/1★ : "
          f"{(d['rating'] == 5).sum() / max((d['rating'] == 1).sum(), 1):.1f}")

    # b) tableau par agence
    rows = []
    for aid, g in d.groupby("agence_id"):
        n = len(g)
        r = g["rating"]
        c = r.value_counts().reindex(stars).fillna(0)
        sd = r.std(ddof=1) if n > 1 else np.nan
        rows.append({
            "agence_id": aid, "n_scrapes": n, "note_moyenne": r.mean(),
            "ic95_demi_largeur": 1.96 * sd / np.sqrt(n) if n > 1 else np.nan,
            **{f"pct_{s}": c[s] / n for s in stars},
            "pct_extremes": (c[1] + c[5]) / n,
            "ratio_5_sur_1": c[5] / max(c[1], 1),
            "pct_avec_texte": g["has_text"].mean(),
            "pct_avec_reponse": g["has_response"].mean(),
            "faible_volume": n < MIN_REVIEWS_PER_AGENCY,
        })
    t = pd.DataFrame(rows).merge(
        ag[["agence_id", "label", "total_fiche", "rating_fiche"]], on="agence_id", how="left"
    )
    t["couverture_scrapes_sur_total"] = t["n_scrapes"] / t["total_fiche"]
    t["ecart_note_scrapee_vs_fiche"] = t["note_moyenne"] - t["rating_fiche"]
    t = t.sort_values("n_scrapes", ascending=False).round(3)
    save_csv(t, "3_biais_par_agence.csv")
    print(t[["label", "n_scrapes", "total_fiche", "couverture_scrapes_sur_total", "note_moyenne",
             "rating_fiche", "pct_extremes", "faible_volume"]].to_string(index=False))

    # c) effet "extrêmes" : qui écrit un texte / qui reçoit une réponse / longueur du texte
    by_star = d.groupby("rating").agg(
        n=("rating", "size"),
        pct_avec_texte=("has_text", "mean"),
        pct_avec_reponse=("has_response", "mean"),
        longueur_texte_moyenne=("text_len", lambda x: x[x > 0].mean()),
    ).round(3)
    save_csv(by_star.reset_index(), "3_biais_par_note.csv")
    print("\nPar note :\n" + by_star.to_string())

    # d) ancienneté (dates relatives approximatives)
    d["anciennete"] = pd.cut(d["age_months"], [-1, 3, 12, 24, 10_000],
                             labels=["≤3 mois", "3-12 mois", "1-2 ans", ">2 ans"])
    by_age = d.groupby("anciennete", observed=True).agg(
        n=("rating", "size"), note_moyenne=("rating", "mean"),
        pct_extremes=("rating", lambda x: x.isin([1, 5]).mean()),
    ).round(3)
    save_csv(by_age.reset_index(), "3_biais_anciennete.csv")
    print("\nPar ancienneté :\n" + by_age.to_string())

    # e) tests statistiques entre agences
    if stats is not None and d["agence_id"].nunique() >= 2:
        ct = pd.crosstab(d["agence_id"], d["rating"])
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        low_exp = (expected < 5).mean()
        groups = [g["rating"].values for _, g in d.groupby("agence_id") if len(g) >= 5]
        kw_p = stats.kruskal(*groups).pvalue if len(groups) >= 2 else np.nan
        print(f"\nTest du chi² (distribution des notes selon l'agence) : p = {p:.4f}, "
              f"V de Cramér = {v:.2f}"
              + (f" | attention : {low_exp:.0%} des effectifs théoriques < 5" if low_exp > 0.2 else ""))
        print(f"Test de Kruskal-Wallis (notes entre agences) : p = {kw_p:.4f}")

    # Graphiques
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(stars, dist.values * 100, color=["#d9534f", "#f0ad4e", "#cccccc", "#9ccc65", "#5cb85c"])
    for s, v in zip(stars, dist.values):
        ax.text(s, v * 100 + 0.5, f"{v:.0%}", ha="center")
    ax.set(title="Distribution des notes (tous avis)", xlabel="étoiles", ylabel="% des avis")
    fig.tight_layout()
    save_fig(fig, "3_notes_distribution_globale.png")

    pct = t.set_index("label")[[f"pct_{s}" for s in stars]] * 100
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(pct) + 2))
    pct.plot.barh(stacked=True, ax=ax,
                  color=["#d9534f", "#f0ad4e", "#cccccc", "#9ccc65", "#5cb85c"])
    ax.legend(["1★", "2★", "3★", "4★", "5★"], loc="lower right")
    ax.set(title="Distribution des notes par agence", xlabel="% des avis", ylabel="")
    fig.tight_layout()
    save_fig(fig, "3_notes_par_agence.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(t["n_scrapes"], t["note_moyenne"], yerr=t["ic95_demi_largeur"],
                fmt="o", color="#4a7fb5", capsize=3)
    ax.axvline(MIN_REVIEWS_PER_AGENCY, color="grey", ls="--")
    ax.set(title="Volume d'avis vs note moyenne (barres : IC à 95 %)",
           xlabel="nombre d'avis scrapés", ylabel="note moyenne")
    fig.tight_layout()
    save_fig(fig, "3_volume_vs_note.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(by_star.index, by_star["pct_avec_texte"] * 100, color="#4a7fb5")
    ax[0].set(title="% d'avis avec commentaire écrit", xlabel="étoiles", ylabel="%")
    ax[1].bar(by_star.index, by_star["pct_avec_reponse"] * 100, color="#8e6bb5")
    ax[1].set(title="% d'avis avec réponse de l'agence", xlabel="étoiles", ylabel="%")
    fig.tight_layout()
    save_fig(fig, "3_biais_texte_reponse.png")

    print(
        "\nPoints d'attention :\n"
        " - Les avis scrapés sont les plus récents (limités par MAX_REVIEWS), pas un échantillon aléatoire :\n"
        "   regarde la colonne 'couverture_scrapes_sur_total'.\n"
        " - Les clients très satisfaits ou très mécontents écrivent plus que les autres (auto-sélection) :\n"
        "   la note Google n'est pas représentative de l'ensemble de la clientèle.\n"
        " - Les agences avec peu d'avis ont des moyennes très instables (voir intervalles de confiance)."
    )


# --------------------------------------------------------------------------- #
# Utilitaires de sortie
# --------------------------------------------------------------------------- #
def save_csv(df, name):
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / name, sep=";", index=False, encoding="utf-8-sig", decimal=",")


def save_fig(fig, name):
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=130)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df, ag = load_data(INPUT_JSONL)
    print(f"{len(ag)} agences, {len(df)} avis dont {int(df['has_text'].sum())} avec texte.")

    df = add_sentiment(df)
    sentiment_outputs(df, ag)
    topic_outputs(df)
    bias_outputs(df, ag)
    print(f"\nRésultats dans : {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
