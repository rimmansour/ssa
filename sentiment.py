import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np

from genai_api_manager.genai_api_client import GenAiApiClient


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "avis_agences.jsonl"
OUTPUT_FILE = "avis_sentiment.jsonl"

MODEL = "gpt-5.6-luna"

# Nombre d'avis envoyés au modèle par requête
BATCH_SIZE = 20

# Température basse pour avoir des classifications plus stables
TEMPERATURE = 0

# Nombre maximum de tentatives en cas d'erreur API / JSON
MAX_RETRIES = 3

# Limite de coût journalière
DAILY_COST_LIMIT = 30

# Si True, les avis déjà présents dans OUTPUT_FILE ne sont pas retraités
RESUME = True


# ============================================================
# CLIENT API
# ============================================================

genai_api_client = GenAiApiClient(
    daily_cost_limit=DAILY_COST_LIMIT
)


# ============================================================
# TAXONOMIE
# ============================================================

VALID_POLARITIES = {
    "positive",
    "negative",
    "neutral",
    "mixed",
    "no_text",
}

VALID_ASPECT_SENTIMENTS = {
    "positive",
    "negative",
    "neutral",
}


# ============================================================
# PROMPT SYSTEME
# ============================================================

SYSTEM_PROMPT = """
Tu es un système expert d'analyse de sentiments de commentaires clients
en français, spécialisé dans les avis Google concernant des agences bancaires.

Ta tâche consiste à analyser le TEXTE de chaque avis.

Tu dois distinguer soigneusement les catégories suivantes :

1. positive
   L'avis est principalement positif.
   Il ne contient pas de critique négative substantielle.

2. negative
   L'avis est principalement négatif.
   Il ne contient pas d'élément positif substantiel.

3. neutral
   L'avis ne contient pas d'opinion positive ou négative substantielle.
   Il est principalement factuel, descriptif ou indique une expérience
   simplement normale/correcte.

   Exemples :
   - "Traitement normal et agence ok"
   - "Je suis venu déposer un chèque, tout s'est bien passé."
   - "Agence correcte, rien à signaler."
   - "J'ai effectué une opération sur mon compte."

4. mixed
   L'avis contient à la fois un élément positif SUBSTANTIEL et un élément
   négatif SUBSTANTIEL.

   Cette catégorie est très importante.

   Elle NE signifie PAS :
   - "sentiment moyen"
   - "avis peu clair"
   - "avis légèrement positif"
   - "avis légèrement négatif"

   Elle signifie qu'il existe réellement des éléments positifs ET négatifs
   qui portent sur l'expérience du client.

   Exemples :
   - "Personnel sympathique mais beaucoup d'attente."
   - "Très bon conseiller, mais les délais sont beaucoup trop longs."
   - "Accueil agréable mais agence souvent difficile à joindre."

5. no_text
   Le champ texte est vide, absent ou ne contient pas suffisamment de texte
   pour déterminer un sentiment.

IMPORTANT :
- Ne déduis JAMAIS le sentiment du texte uniquement à partir de la note Google.
- La note Google est fournie comme information contextuelle mais le texte
  doit rester la source principale pour déterminer le sentiment.
- Un avis noté 5 étoiles peut être négatif si le texte l'est.
- Un avis noté 1 étoile peut être sarcastique ou ironique.
- Détecte le sarcasme et l'ironie.

Exemple :
"Je suis tellement content de votre banque que je vous ai mis une étoile."
=> negative
car la formulation est sarcastique.

============================================================
INTENSITE
============================================================

Retourne une intensité entre 0 et 1.

L'intensité mesure la force de l'opinion exprimée.

0.0 = aucune opinion / très factuel
0.2 = opinion très faible
0.4 = opinion faible
0.6 = opinion claire
0.8 = opinion forte
1.0 = opinion extrêmement forte

L'intensité ne dépend PAS de la longueur du texte.

Exemples :

"Agence correcte."
=> intensité faible

"Personnel très sympathique."
=> intensité moyenne/forte

"Service absolument catastrophique, une honte."
=> intensité très forte

============================================================
CONFIDENCE
============================================================

Retourne une confiance entre 0 et 1.

Elle représente ton niveau de confiance dans la classification.

1.0 = classification très claire
0.8 = classification claire
0.5 = ambiguïté notable
0.2 = très incertain

============================================================
EVIDENCES
============================================================

Extrais les éléments textuels qui justifient le sentiment.

positive_evidence :
liste de courts extraits ou reformulations très proches du texte
qui correspondent à des éléments positifs.

negative_evidence :
liste de courts extraits ou reformulations très proches du texte
qui correspondent à des éléments négatifs.

Pour un avis neutral :
les deux listes doivent généralement être vides.

Pour un avis positive :
positive_evidence doit normalement contenir au moins un élément.

Pour un avis negative :
negative_evidence doit normalement contenir au moins un élément.

Pour un avis mixed :
positive_evidence ET negative_evidence doivent normalement être non vides.

============================================================
ASPECTS
============================================================

Identifie les aspects importants mentionnés dans l'avis.

Pour chaque aspect, retourne :

- aspect : nom court et normalisé
- sentiment : positive / negative / neutral
- intensity : nombre entre 0 et 1

Exemples d'aspects bancaires :

accueil
personnel
conseiller
professionnalisme
disponibilite
ecoute
reactivite
telephone
attente
delais
operations_bancaires
carte_bancaire
distributeur
horaires
credit
compte
frais
application
directeur_agence
service_client
agence
accessibilite
localisation

Tu peux créer un autre aspect si nécessaire.

IMPORTANT :
Un même avis peut avoir plusieurs aspects avec des sentiments différents.

Exemple :
"Le personnel est très sympathique mais il y a beaucoup d'attente."

=> mixed

aspects :
[
  {
    "aspect": "personnel",
    "sentiment": "positive",
    "intensity": 0.8
  },
  {
    "aspect": "attente",
    "sentiment": "negative",
    "intensity": 0.8
  }
]

============================================================
REGLES IMPORTANTES POUR NEUTRAL VS MIXED
============================================================

"Agence correcte, rien à signaler."
=> neutral

"Personnel sympathique mais beaucoup d'attente."
=> mixed

"Bon accueil, agence proche de chez moi."
=> positive

La proximité est ici une information contextuelle et pas nécessairement
un élément négatif.

"Très bon accueil, même si l'agence est petite."
=> mixed SEULEMENT si "petite" est clairement présenté comme une critique.

Ne transforme pas automatiquement toute opposition grammaticale
("mais", "cependant", "même si", etc.) en mixed.

Il faut qu'il existe réellement un élément positif ET un élément négatif.

============================================================
SORTIE
============================================================

Retourne UNIQUEMENT un tableau JSON valide.

Aucun texte avant ou après le JSON.

Format exact :

[
  {
    "id": "ID_DE_L_AVIS",
    "polarity": "positive",
    "intensity": 0.75,
    "confidence": 0.95,
    "positive_evidence": [
      "personnel très sympathique"
    ],
    "negative_evidence": [],
    "aspects": [
      {
        "aspect": "personnel",
        "sentiment": "positive",
        "intensity": 0.8
      }
    ]
  }
]

Pour no_text :

[
  {
    "id": "ID_DE_L_AVIS",
    "polarity": "no_text",
    "intensity": 0.0,
    "confidence": 1.0,
    "positive_evidence": [],
    "negative_evidence": [],
    "aspects": []
  }
]

Ne retourne AUCUN champ supplémentaire.
"""


# ============================================================
# UTILITAIRES JSONL
# ============================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Charge un fichier JSONL.
    Une ligne = un objet JSON.
    """
    data = []

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"[WARNING] Ligne {line_number} ignorée : "
                    f"JSON invalide ({e})"
                )

    return data


def append_jsonl(
    path: str,
    rows: List[Dict[str, Any]]
) -> None:
    """
    Ajoute des objets dans un fichier JSONL.
    """
    with open(path, "a", encoding="utf-8") as f:

        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# NORMALISATION DU TEXTE
# ============================================================

def clean_text(text: Any) -> str:
    """
    Nettoie légèrement le texte sans supprimer les informations
    importantes pour l'analyse du sentiment.
    """

    if text is None:
        return ""

    text = str(text)

    # Remplace les espaces multiples
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# CHARGEMENT DES AVIS
# ============================================================

def flatten_reviews(
    agencies: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Transforme :

    agence -> liste reviews

    en :

    une ligne = un avis
    """

    reviews = []

    for agency in agencies:

        place = agency.get("place") or {}

        agency_id = agency.get("id")

        agency_name = place.get("name")
        agency_address = place.get("address")
        agency_rating = place.get("rating")
        agency_total_reviews = place.get("total_reviews")
        agency_url = place.get("url")
        agency_lat = place.get("lat")
        agency_lng = place.get("lng")

        raw_reviews = agency.get("reviews") or []

        for review_index, review in enumerate(
            raw_reviews
        ):

            text = clean_text(
                review.get("text")
            )

            review_id = review.get(
                "review_id"
            )

            # Certains avis peuvent ne pas avoir de review_id.
            # On crée alors un identifiant déterministe.
            if not review_id:
                review_id = (
                    f"{agency_id}_review_{review_index}"
                )

            row = {
                "id": review_id,

                "agency_id": agency_id,
                "agency_name": agency_name,
                "agency_address": agency_address,
                "agency_rating": agency_rating,
                "agency_total_reviews": agency_total_reviews,
                "agency_url": agency_url,
                "agency_lat": agency_lat,
                "agency_lng": agency_lng,

                "author": review.get(
                    "author"
                ),

                "author_info": review.get(
                    "author_info"
                ),

                # Note Google donnée par l'utilisateur
                "google_rating": review.get(
                    "rating"
                ),

                "date": review.get(
                    "date"
                ),

                "text": text,

                "owner_response": review.get(
                    "owner_response"
                ),

                "owner_response_date": review.get(
                    "owner_response_date"
                ),
            }

            reviews.append(row)

    return reviews


# ============================================================
# EXTRACTION JSON DE LA REPONSE LLM
# ============================================================

def extract_json(
    content: str
) -> Any:
    """
    Essaie d'extraire un JSON même si le modèle ajoute accidentellement
    des ```json ... ```.
    """

    if not content:
        raise ValueError(
            "Réponse LLM vide"
        )

    content = content.strip()

    # Cas normal
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Supprime les fences markdown
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        content,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Recherche du premier tableau JSON
    start = cleaned.find("[")
    end = cleaned.rfind("]")

    if start != -1 and end != -1 and end > start:

        candidate = cleaned[
            start:end + 1
        ]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Impossible d'extraire un JSON valide "
        f"depuis la réponse : {content[:500]}"
    )


# ============================================================
# VALIDATION D'UNE REPONSE
# ============================================================

def clamp_float(
    value: Any,
    default: float = 0.0
) -> float:

    try:
        value = float(value)
    except (
        TypeError,
        ValueError
    ):
        return default

    return max(
        0.0,
        min(1.0, value)
    )


def validate_result(
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Valide et normalise un résultat LLM.
    """

    if not isinstance(
        result,
        dict
    ):
        raise ValueError(
            "Un résultat doit être un objet JSON."
        )

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    if "id" not in result:
        raise ValueError(
            "Champ 'id' absent."
        )

    result["id"] = str(
        result["id"]
    )

    # --------------------------------------------------------
    # POLARITY
    # --------------------------------------------------------

    polarity = result.get(
        "polarity"
    )

    if polarity not in VALID_POLARITIES:

        raise ValueError(
            f"Polarity invalide : {polarity}"
        )

    result["polarity"] = polarity

    # --------------------------------------------------------
    # INTENSITY
    # --------------------------------------------------------

    result["intensity"] = clamp_float(
        result.get("intensity"),
        default=0.0
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    result["confidence"] = clamp_float(
        result.get("confidence"),
        default=0.0
    )

    # --------------------------------------------------------
    # EVIDENCES
    # --------------------------------------------------------

    positive_evidence = result.get(
        "positive_evidence",
        []
    )

    negative_evidence = result.get(
        "negative_evidence",
        []
    )

    if not isinstance(
        positive_evidence,
        list
    ):
        positive_evidence = []

    if not isinstance(
        negative_evidence,
        list
    ):
        negative_evidence = []

    result["positive_evidence"] = [
        str(x)
        for x in positive_evidence
        if x is not None
    ]

    result["negative_evidence"] = [
        str(x)
        for x in negative_evidence
        if x is not None
    ]

    # --------------------------------------------------------
    # ASPECTS
    # --------------------------------------------------------

    raw_aspects = result.get(
        "aspects",
        []
    )

    if not isinstance(
        raw_aspects,
        list
    ):
        raw_aspects = []

    aspects = []

    for aspect in raw_aspects:

        if not isinstance(
            aspect,
            dict
        ):
            continue

        aspect_name = aspect.get(
            "aspect"
        )

        sentiment = aspect.get(
            "sentiment"
        )

        if not aspect_name:
            continue

        if sentiment not in VALID_ASPECT_SENTIMENTS:
            continue

        aspects.append({
            "aspect": str(
                aspect_name
            ).strip(),

            "sentiment": sentiment,

            "intensity": clamp_float(
                aspect.get(
                    "intensity"
                ),
                default=0.0
            )
        })

    result["aspects"] = aspects

    # --------------------------------------------------------
    # COHERENCE CHECKS
    # --------------------------------------------------------

    if polarity == "no_text":

        result["intensity"] = 0.0
        result["positive_evidence"] = []
        result["negative_evidence"] = []
        result["aspects"] = []

    return result


# ============================================================
# CALCUL DU BALANCE
# ============================================================

def calculate_balance(
    aspects: List[Dict[str, Any]]
) -> float:
    """
    Calcule un score de balance à partir des sentiments
    par aspect.

    positive => +intensity
    negative => -intensity
    neutral  => 0

    Résultat approximativement entre -1 et +1.
    """

    scores = []

    for aspect in aspects:

        sentiment = aspect.get(
            "sentiment"
        )

        intensity = clamp_float(
            aspect.get(
                "intensity"
            ),
            default=0.0
        )

        if sentiment == "positive":

            scores.append(
                intensity
            )

        elif sentiment == "negative":

            scores.append(
                -intensity
            )

        elif sentiment == "neutral":

            scores.append(
                0.0
            )

    if not scores:
        return 0.0

    return float(
        sum(scores) / len(scores)
    )


# ============================================================
# CONTROLE DE COHERENCE SUPPLEMENTAIRE
# ============================================================

def enforce_consistency(
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Applique quelques règles déterministes après la réponse LLM.

    On ne remplace pas le jugement du modèle :
    on corrige seulement les incohérences évidentes.
    """

    polarity = result["polarity"]

    positive_evidence = result[
        "positive_evidence"
    ]

    negative_evidence = result[
        "negative_evidence"
    ]

    aspects = result[
        "aspects"
    ]

    # --------------------------------------------------------
    # no_text
    # --------------------------------------------------------

    if polarity == "no_text":

        result["intensity"] = 0.0
        result["balance"] = 0.0

        return result

    # --------------------------------------------------------
    # BALANCE
    # --------------------------------------------------------

    result["balance"] = round(
        calculate_balance(
            aspects
        ),
        4
    )

    # --------------------------------------------------------
    # Si le modèle dit mixed mais qu'il n'existe clairement
    # qu'une seule polarité dans les aspects/evidences,
    # on conserve toutefois mixed car les aspects peuvent
    # manquer une information implicite.
    #
    # On évite donc une correction agressive.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Cohérence neutral
    # --------------------------------------------------------

    if polarity == "neutral":

        # Un avis neutral ne devrait normalement pas avoir
        # d'évidence positive/négative forte.
        #
        # On ne change PAS la classe automatiquement :
        # on garde la décision LLM pour éviter les faux positifs.
        pass

    # --------------------------------------------------------
    # Cohérence mixed
    # --------------------------------------------------------

    if polarity == "mixed":

        has_positive = (
            len(positive_evidence) > 0
            or any(
                a["sentiment"] == "positive"
                for a in aspects
            )
        )

        has_negative = (
            len(negative_evidence) > 0
            or any(
                a["sentiment"] == "negative"
                for a in aspects
            )
        )

        # On conserve mixed même si l'un des éléments
        # est implicite, mais on pourrait ici ajouter
        # un flag de qualité.
        result["mixed_evidence_complete"] = (
            has_positive and has_negative
        )

    return result


# ============================================================
# PROMPT D'UN BATCH
# ============================================================

def build_batch_prompt(
    batch: List[Dict[str, Any]]
) -> str:
    """
    Construit le prompt utilisateur pour un lot d'avis.
    """

    simplified_reviews = []

    for review in batch:

        simplified_reviews.append({
            "id": review["id"],
            "google_rating": review.get(
                "google_rating"
            ),
            "date": review.get(
                "date"
            ),
            "text": review.get(
                "text",
                ""
            ),
        })

    payload = json.dumps(
        simplified_reviews,
        ensure_ascii=False,
        indent=2
    )

    prompt = f"""
Analyse les avis suivants.

Pour chaque avis :
- utilise son id exact
- analyse le texte
- ne déduis pas mécaniquement le sentiment depuis google_rating
- distingue soigneusement neutral et mixed
- détecte le sarcasme
- extrais les aspects
- donne le sentiment de chaque aspect
- donne les evidences positives et négatives

Retourne uniquement le tableau JSON demandé.

AVIS :

{payload}
"""

    return prompt


# ============================================================
# APPEL LLM
# ============================================================

def analyze_batch(
    batch: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    prompt = build_batch_prompt(
        batch
    )

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = (
                genai_api_client
                .get_genai_completion(
                    system_message=SYSTEM_PROMPT,
                    prompt=prompt,
                    model=MODEL,
                    temperature=TEMPERATURE,
                )
            )

            content = (
                response
                .choices[0]
                .message
                .content
            )

            parsed = extract_json(
                content
            )

            if not isinstance(
                parsed,
                list
            ):
                raise ValueError(
                    "La réponse JSON n'est pas une liste."
                )

            results = []

            for item in parsed:

                validated = validate_result(
                    item
                )

                validated = enforce_consistency(
                    validated
                )

                results.append(
                    validated
                )

            return results

        except Exception as e:

            last_error = e

            print(
                f"[WARNING] Tentative "
                f"{attempt}/{MAX_RETRIES} échouée : "
                f"{e}"
            )

            if attempt < MAX_RETRIES:

                # Backoff progressif
                sleep_time = (
                    2 ** (attempt - 1)
                )

                time.sleep(
                    sleep_time
                )

    raise RuntimeError(
        f"Echec analyse batch après "
        f"{MAX_RETRIES} tentatives : "
        f"{last_error}"
    )


# ============================================================
# CREATION D'UN INDEX DES AVIS DEJA TRAITES
# ============================================================

def load_processed_ids(
    output_file: str
) -> set:

    processed_ids = set()

    if not os.path.exists(
        output_file
    ):
        return processed_ids

    with open(
        output_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                row = json.loads(
                    line
                )

                review_id = row.get(
                    "id"
                )

                if review_id is not None:

                    processed_ids.add(
                        str(review_id)
                    )

            except json.JSONDecodeError:

                continue

    return processed_ids


# ============================================================
# CONSTRUCTION DU RESULTAT FINAL
# ============================================================

def merge_analysis_with_review(
    review: Dict[str, Any],
    analysis: Dict[str, Any]
) -> Dict[str, Any]:

    result = dict(
        review
    )

    result.update({
        "polarity": analysis[
            "polarity"
        ],

        "intensity": analysis[
            "intensity"
        ],

        "confidence": analysis[
            "confidence"
        ],

        "positive_evidence": analysis[
            "positive_evidence"
        ],

        "negative_evidence": analysis[
            "negative_evidence"
        ],

        "aspects": analysis[
            "aspects"
        ],

        "balance": analysis.get(
            "balance",
            0.0
        ),
    })

    # Champ utile pour contrôler mixed
    if "mixed_evidence_complete" in analysis:

        result[
            "mixed_evidence_complete"
        ] = analysis[
            "mixed_evidence_complete"
        ]

    return result


# ============================================================
# ANALYSE PRINCIPALE
# ============================================================

def run():

    print("=" * 70)
    print("CHARGEMENT DES DONNEES")
    print("=" * 70)

    agencies = load_jsonl(
        INPUT_FILE
    )

    print(
        f"Agences chargées : {len(agencies)}"
    )

    reviews = flatten_reviews(
        agencies
    )

    print(
        f"Avis trouvés : {len(reviews)}"
    )

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    if RESUME:

        processed_ids = load_processed_ids(
            OUTPUT_FILE
        )

        print(
            f"Avis déjà traités : "
            f"{len(processed_ids)}"
        )

    else:

        processed_ids = set()

        # Si on recommence depuis zéro,
        # on supprime le fichier de sortie.
        if os.path.exists(
            OUTPUT_FILE
        ):
            os.remove(
                OUTPUT_FILE
            )

    # --------------------------------------------------------
    # AVIS A TRAITER
    # --------------------------------------------------------

    reviews_to_process = [
        review
        for review in reviews
        if str(
            review["id"]
        ) not in processed_ids
    ]

    print(
        f"Avis restant à analyser : "
        f"{len(reviews_to_process)}"
    )

    # --------------------------------------------------------
    # AVIS SANS TEXTE
    #
    # Pas besoin d'appeler le LLM.
    # --------------------------------------------------------

    no_text_reviews = []

    reviews_with_text = []

    for review in reviews_to_process:

        text = clean_text(
            review.get(
                "text"
            )
        )

        if not text:

            no_text_reviews.append(
                review
            )

        else:

            reviews_with_text.append(
                review
            )

    # --------------------------------------------------------
    # ECRITURE DIRECTE DES no_text
    # --------------------------------------------------------

    if no_text_reviews:

        print(
            f"Avis sans texte : "
            f"{len(no_text_reviews)}"
        )

        no_text_results = []

        for review in no_text_reviews:

            analysis = {
                "id": review["id"],
                "polarity": "no_text",
                "intensity": 0.0,
                "confidence": 1.0,
                "positive_evidence": [],
                "negative_evidence": [],
                "aspects": [],
                "balance": 0.0,
            }

            final_row = merge_analysis_with_review(
                review,
                analysis
            )

            no_text_results.append(
                final_row
            )

        append_jsonl(
            OUTPUT_FILE,
            no_text_results
        )

        print(
            f"{len(no_text_results)} avis no_text sauvegardés."
        )

    # --------------------------------------------------------
    # ANALYSE PAR BATCH
    # --------------------------------------------------------

    total = len(
        reviews_with_text
    )

    print(
        f"Avis avec texte à analyser : {total}"
    )

    if total == 0:

        print(
            "Aucun avis textuel à analyser."
        )

        return

    number_of_batches = (
        total + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for batch_index in range(
        number_of_batches
    ):

        start = (
            batch_index * BATCH_SIZE
        )

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch = reviews_with_text[
            start:end
        ]

        print()
        print(
            "=" * 70
        )
        print(
            f"BATCH "
            f"{batch_index + 1}/"
            f"{number_of_batches}"
        )
        print(
            f"Avis {start + 1} -> {end}"
        )
        print(
            "=" * 70
        )

        try:

            analyses = analyze_batch(
                batch
            )

        except Exception as e:

            print(
                f"[ERROR] Batch "
                f"{batch_index + 1} "
                f"abandonné : {e}"
            )

            # On arrête ici plutôt que de produire
            # un fichier partiellement incohérent.
            raise

        # ----------------------------------------------------
        # INDEX PAR ID
        # ----------------------------------------------------

        analyses_by_id = {}

        for analysis in analyses:

            analyses_by_id[
                str(analysis["id"])
            ] = analysis

        # ----------------------------------------------------
        # VERIFICATION
        # ----------------------------------------------------

        expected_ids = {
            str(review["id"])
            for review in batch
        }

        returned_ids = set(
            analyses_by_id.keys()
        )

        missing_ids = (
            expected_ids - returned_ids
        )

        extra_ids = (
            returned_ids - expected_ids
        )

        if missing_ids:

            raise ValueError(
                "Le modèle n'a pas retourné "
                f"les avis suivants : "
                f"{list(missing_ids)[:10]}"
            )

        if extra_ids:

            print(
                "[WARNING] IDs inattendus retournés :",
                list(extra_ids)[:10]
            )

        # ----------------------------------------------------
        # MERGE
        # ----------------------------------------------------

        output_rows = []

        for review in batch:

            review_id = str(
                review["id"]
            )

            analysis = analyses_by_id[
                review_id
            ]

            final_row = merge_analysis_with_review(
                review,
                analysis
            )

            output_rows.append(
                final_row
            )

        # ----------------------------------------------------
        # SAUVEGARDE IMMEDIATE
        #
        # Très important :
        # si le script plante au batch suivant,
        # tout ce qui précède est déjà sauvegardé.
        # ----------------------------------------------------

        append_jsonl(
            OUTPUT_FILE,
            output_rows
        )

        print(
            f"[OK] {len(output_rows)} avis sauvegardés."
        )

    print()
    print(
        "=" * 70
    )
    print(
        "ANALYSE TERMINEE"
    )
    print(
        "=" * 70
    )

    print(
        f"Fichier de sortie : {OUTPUT_FILE}"
    )


# ============================================================
# STATISTIQUES POST-TRAITEMENT
# ============================================================

def compute_statistics(
    output_file: str = OUTPUT_FILE
):

    if not os.path.exists(
        output_file
    ):

        print(
            "Fichier de sortie introuvable."
        )

        return

    rows = load_jsonl(
        output_file
    )

    print()
    print(
        "=" * 70
    )
    print(
        "STATISTIQUES"
    )
    print(
        "=" * 70
    )

    print(
        f"Nombre total d'avis : {len(rows)}"
    )

    # --------------------------------------------------------
    # POLARITES
    # --------------------------------------------------------

    polarity_counts = {}

    for row in rows:

        polarity = row.get(
            "polarity",
            "unknown"
        )

        polarity_counts[
            polarity
        ] = (
            polarity_counts.get(
                polarity,
                0
            )
            + 1
        )

    print()
    print(
        "Répartition des sentiments :"
    )

    for polarity, count in sorted(
        polarity_counts.items()
    ):

        percentage = (
            count / len(rows) * 100
            if rows
            else 0
        )

        print(
            f"  {polarity:10s} : "
            f"{count:6d} "
            f"({percentage:5.1f}%)"
        )

    # --------------------------------------------------------
    # SENTIMENT PAR NOTE GOOGLE
    # --------------------------------------------------------

    print()
    print(
        "Sentiment textuel par note Google :"
    )

    matrix = {}

    for row in rows:

        rating = row.get(
            "google_rating"
        )

        polarity = row.get(
            "polarity"
        )

        if rating is None:
            rating = "NA"

        if rating not in matrix:
            matrix[rating] = {}

        matrix[rating][
            polarity
        ] = (
            matrix[rating].get(
                polarity,
                0
            )
            + 1
        )

    for rating in sorted(
        matrix.keys(),
        key=lambda x: str(x)
    ):

        print(
            f"  Note {rating}:"
        )

        for polarity, count in sorted(
            matrix[rating].items()
        ):

            print(
                f"      {polarity:10s}: "
                f"{count}"
            )

    # --------------------------------------------------------
    # BALANCE
    # --------------------------------------------------------

    balances = []

    for row in rows:

        balance = row.get(
            "balance"
        )

        if isinstance(
            balance,
            (int, float)
        ):

            balances.append(
                balance
            )

    if balances:

        print()
        print(
            "Balance moyen : "
            f"{np.mean(balances):.3f}"
        )

        print(
            "Balance médiane : "
            f"{np.median(balances):.3f}"
        )

    # --------------------------------------------------------
    # TOP ASPECTS
    # --------------------------------------------------------

    aspect_counts = {}

    for row in rows:

        for aspect in row.get(
            "aspects",
            []
        ):

            name = aspect.get(
                "aspect"
            )

            if not name:
                continue

            if name not in aspect_counts:

                aspect_counts[name] = {
                    "total": 0,
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                }

            aspect_counts[name][
                "total"
            ] += 1

            sentiment = aspect.get(
                "sentiment"
            )

            if sentiment in {
                "positive",
                "negative",
                "neutral"
            }:

                aspect_counts[name][
                    sentiment
                ] += 1

    print()
    print(
        "Principaux aspects :"
    )

    sorted_aspects = sorted(
        aspect_counts.items(),
        key=lambda x: x[1]["total"],
        reverse=True
    )

    for aspect, stats in sorted_aspects[:20]:

        print(
            f"  {aspect:25s} "
            f"total={stats['total']:5d} "
            f"+={stats['positive']:5d} "
            f"-={stats['negative']:5d} "
            f"neutral={stats['neutral']:5d}"
        )


# ============================================================
# POINT D'ENTREE
# ============================================================

if __name__ == "__main__":

    run()

    # Décommente si tu veux afficher les statistiques
    # automatiquement après l'analyse.
    compute_statistics()