import asyncio
import os
import re
import time
from datetime import datetime

import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "agences_sg.xlsx"
OUTPUT_FILE = "avis_sg.xlsx"

AGENCE_COL = "agence"
ADRESSE_COL = "adresse"

# Nombre maximum d'avis recherchés par agence.
# Mets None pour essayer d'aller jusqu'à la fin.
MAX_REVIEWS = 500

# Temps d'attente entre certaines actions
MIN_WAIT = 1.0
MAX_WAIT = 2.5

# Mode headless :
# False = tu vois le navigateur (recommandé pour le premier test)
# True  = navigateur invisible
HEADLESS = False

# Nombre maximum de scrolls dans la fenêtre des avis
MAX_SCROLLS = 150

# ============================================================
# OUTILS
# ============================================================

def clean_text(text):
    """Nettoie un texte."""
    if not text:
        return ""

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


async def random_wait(page, min_wait=MIN_WAIT, max_wait=MAX_WAIT):
    """Petite pause entre les actions."""
    import random

    await page.wait_for_timeout(
        int(random.uniform(min_wait, max_wait) * 1000)
    )


async def click_if_exists(page, selectors, timeout=3000):
    """
    Essaie plusieurs sélecteurs et clique sur le premier trouvé.
    """
    for selector in selectors:
        try:
            locator = page.locator(selector).first

            if await locator.count() > 0:
                await locator.click(timeout=timeout)
                return True

        except Exception:
            pass

    return False


# ============================================================
# RECHERCHE GOOGLE MAPS
# ============================================================

async def search_google_maps(page, address):
    """
    Recherche une adresse sur Google Maps.
    """

    search_url = (
        "https://www.google.com/maps/search/"
        + address.replace(" ", "+")
    )

    await page.goto(
        search_url,
        wait_until="domcontentloaded",
        timeout=60000
    )

    await page.wait_for_timeout(3000)

    return search_url


# ============================================================
# ACCEPTATION DES COOKIES
# ============================================================

async def accept_cookies(page):
    """
    Essaie d'accepter les cookies si Google les demande.
    """

    selectors = [
        'button:has-text("Tout accepter")',
        'button:has-text("Accepter tout")',
        'button:has-text("Accept all")',
        'button:has-text("I agree")',
    ]

    await click_if_exists(page, selectors, timeout=3000)


# ============================================================
# VERIFICATION SOCIETE GENERALE
# ============================================================

async def verify_societe_generale(page):
    """
    Vérifie grossièrement que la page correspond à Société Générale.

    Cette vérification n'est pas parfaite : Google peut afficher
    plusieurs résultats dans certaines recherches.
    """

    try:
        body_text = await page.locator("body").inner_text(timeout=5000)

        body_text_lower = body_text.lower()

        keywords = [
            "société générale",
            "societe generale",
            "société générale",
        ]

        return any(keyword in body_text_lower for keyword in keywords)

    except Exception:
        return False


# ============================================================
# OUVERTURE DES AVIS
# ============================================================

async def open_reviews(page):
    """
    Essaie d'ouvrir la section des avis.
    """

    selectors = [
        'button:has-text("avis")',
        'button:has-text("Avis")',
        '[aria-label*="avis"]',
        '[aria-label*="Reviews"]',
        'text=Avis',
        'text=reviews',
    ]

    # Première tentative
    clicked = await click_if_exists(
        page,
        selectors,
        timeout=5000
    )

    if clicked:
        await page.wait_for_timeout(2500)
        return True

    # Deuxième méthode :
    # recherche d'un élément contenant le nombre d'avis
    try:
        elements = page.locator(
            'button, div[role="button"], a'
        )

        count = await elements.count()

        for i in range(min(count, 200)):

            try:
                text = clean_text(
                    await elements.nth(i).inner_text(timeout=1000)
                )

                if "avis" in text.lower() or "review" in text.lower():

                    await elements.nth(i).click(
                        timeout=3000
                    )

                    await page.wait_for_timeout(2500)

                    return True

            except Exception:
                continue

    except Exception:
        pass

    return False


# ============================================================
# IDENTIFICATION DU PANNEAU DES AVIS
# ============================================================

async def get_review_scroll_container(page):
    """
    Cherche le conteneur scrollable des avis.

    Google Maps change parfois la structure HTML.
    """

    possible_selectors = [
        'div[role="feed"]',
        'div.m6QErb',
    ]

    for selector in possible_selectors:

        try:
            locator = page.locator(selector).first

            if await locator.count() > 0:

                # Vérification qu'il existe réellement
                box = await locator.bounding_box()

                if box:
                    return locator

        except Exception:
            pass

    return None


# ============================================================
# SCROLL DES AVIS
# ============================================================

async def scroll_reviews(page, max_scrolls=MAX_SCROLLS):

    container = await get_review_scroll_container(page)

    if container is None:
        print("   ⚠️ Conteneur des avis introuvable.")
        return

    previous_height = 0
    unchanged_count = 0

    for scroll_number in range(max_scrolls):

        try:

            current_height = await container.evaluate(
                "(element) => element.scrollHeight"
            )

            await container.evaluate(
                "(element) => element.scrollTo("
                "0, element.scrollHeight)"
            )

            await page.wait_for_timeout(1500)

            new_height = await container.evaluate(
                "(element) => element.scrollHeight"
            )

            # Google peut charger de nouveaux avis
            if new_height == previous_height:

                unchanged_count += 1

            else:

                unchanged_count = 0

            previous_height = new_height

            print(
                f"      Scroll {scroll_number + 1}/{max_scrolls} "
                f"| hauteur={new_height}"
            )

            # Si rien ne change plusieurs fois,
            # on considère qu'on est arrivé au bout.
            if unchanged_count >= 5:
                print("      Fin probable des avis.")
                break

        except Exception as e:

            print(
                f"      ⚠️ Erreur pendant le scroll : {e}"
            )

            break


# ============================================================
# EXTRACTION DES AVIS
# ============================================================

async def extract_reviews(page):
    """
    Extrait les avis visibles dans le DOM.
    """

    reviews = []

    # Les avis Google Maps utilisent généralement
    # des éléments data-review-id.
    review_elements = page.locator(
        '[data-review-id]'
    )

    count = await review_elements.count()

    print(f"      Éléments d'avis détectés : {count}")

    for i in range(count):

        try:

            review = review_elements.nth(i)

            # ------------------------------------------------
            # Auteur
            # ------------------------------------------------

            author = ""

            author_selectors = [
                '.d4r55',
                '[class*="d4r55"]',
            ]

            for selector in author_selectors:

                try:
                    loc = review.locator(selector).first

                    if await loc.count() > 0:

                        author = clean_text(
                            await loc.inner_text(timeout=1000)
                        )

                        if author:
                            break

                except Exception:
                    pass

            # ------------------------------------------------
            # Note
            # ------------------------------------------------

            rating = ""

            rating_selectors = [
                'span[role="img"]',
                '[aria-label*="étoile"]',
                '[aria-label*="star"]',
            ]

            for selector in rating_selectors:

                try:
                    loc = review.locator(selector).first

                    if await loc.count() > 0:

                        rating = (
                            await loc.get_attribute("aria-label")
                        ) or ""

                        if rating:
                            break

                except Exception:
                    pass

            # ------------------------------------------------
            # Date
            # ------------------------------------------------

            date = ""

            date_selectors = [
                '.rsqaWe',
                '[class*="rsqaWe"]',
            ]

            for selector in date_selectors:

                try:
                    loc = review.locator(selector).first

                    if await loc.count() > 0:

                        date = clean_text(
                            await loc.inner_text(timeout=1000)
                        )

                        if date:
                            break

                except Exception:
                    pass

            # ------------------------------------------------
            # Commentaire
            # ------------------------------------------------

            comment = ""

            comment_selectors = [
                '.wiI7pd',
                '[class*="wiI7pd"]',
            ]

            for selector in comment_selectors:

                try:
                    loc = review.locator(selector).first

                    if await loc.count() > 0:

                        comment = clean_text(
                            await loc.inner_text(timeout=1000)
                        )

                        if comment:
                            break

                except Exception:
                    pass

            # ------------------------------------------------
            # ID Google
            # ------------------------------------------------

            review_id = await review.get_attribute(
                "data-review-id"
            )

            # ------------------------------------------------
            # Ajouter seulement si on a réellement quelque chose
            # ------------------------------------------------

            if author or comment or rating:

                reviews.append(
                    {
                        "review_id": review_id,
                        "auteur": author,
                        "note": rating,
                        "date_avis": date,
                        "commentaire": comment,
                    }
                )

        except Exception as e:

            print(
                f"      ⚠️ Erreur extraction avis {i}: {e}"
            )

    return reviews


# ============================================================
# SAUVEGARDE PROGRESSIVE
# ============================================================

def save_results(rows, output_file):

    if not rows:
        return

    df_new = pd.DataFrame(rows)

    if os.path.exists(output_file):

        try:

            df_old = pd.read_excel(output_file)

            df = pd.concat(
                [df_old, df_new],
                ignore_index=True
            )

            # Élimination des doublons
            if "review_id" in df.columns:

                df = df.drop_duplicates(
                    subset=["review_id"],
                    keep="first"
                )

        except Exception:

            df = df_new

    else:

        df = df_new

    df.to_excel(
        output_file,
        index=False
    )


# ============================================================
# TRAITEMENT D'UNE AGENCE
# ============================================================

async def process_agency(
    page,
    agency_name,
    address
):

    print("\n" + "=" * 70)

    print(
        f"Agence : {agency_name}"
    )

    print(
        f"Adresse : {address}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Recherche
    # --------------------------------------------------------

    search_url = await search_google_maps(
        page,
        address
    )

    await accept_cookies(page)

    await random_wait(page)

    # --------------------------------------------------------
    # Vérification SG
    # --------------------------------------------------------

    is_sg = await verify_societe_generale(page)

    if not is_sg:

        print(
            "   ⚠️ Société Générale non détectée."
        )

        return []

    print(
        "   ✓ Société Générale détectée."
    )

    # --------------------------------------------------------
    # URL finale de la page
    # --------------------------------------------------------

    google_maps_url = page.url

    # --------------------------------------------------------
    # Ouvrir les avis
    # --------------------------------------------------------

    opened = await open_reviews(page)

    if not opened:

        print(
            "   ⚠️ Impossible d'ouvrir les avis."
        )

        return []

    print(
        "   ✓ Section avis ouverte."
    )

    # --------------------------------------------------------
    # Scroll
    # --------------------------------------------------------

    await scroll_reviews(
        page,
        MAX_SCROLLS
    )

    # --------------------------------------------------------
    # Extraction
    # --------------------------------------------------------

    reviews = await extract_reviews(page)

    print(
        f"   ✓ {len(reviews)} avis récupérés."
    )

    # --------------------------------------------------------
    # Ajouter métadonnées agence
    # --------------------------------------------------------

    final_rows = []

    for review in reviews:

        review["agence"] = agency_name
        review["adresse"] = address
        review["google_maps_url"] = google_maps_url
        review["scraped_at"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        final_rows.append(review)

    return final_rows


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

async def main():

    # --------------------------------------------------------
    # Lecture Excel
    # --------------------------------------------------------

    print(
        f"Lecture du fichier : {INPUT_FILE}"
    )

    df = pd.read_excel(INPUT_FILE)

    # Vérification des colonnes
    required_columns = [
        AGENCE_COL,
        ADRESSE_COL
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Colonne '{column}' absente du fichier Excel."
            )

    print(
        f"{len(df)} agences trouvées."
    )

    # --------------------------------------------------------
    # Lancement Playwright
    # --------------------------------------------------------

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=HEADLESS,
            args=[
                "--disable-blink-features=AutomationControlled",
            ]
        )

        context = await browser.new_context(
            locale="fr-FR",
            viewport={
                "width": 1440,
                "height": 900
            }
        )

        page = await context.new_page()

        # ----------------------------------------------------
        # Traitement agence par agence
        # ----------------------------------------------------

        for index, row in df.iterrows():

            agency_name = str(
                row[AGENCE_COL]
            ).strip()

            address = str(
                row[ADRESSE_COL]
            ).strip()

            print(
                f"\n[{index + 1}/{len(df)}]"
            )

            try:

                rows = await process_agency(
                    page,
                    agency_name,
                    address
                )

                # --------------------------------------------
                # Sauvegarde IMMÉDIATE
                # --------------------------------------------

                if rows:

                    save_results(
                        rows,
                        OUTPUT_FILE
                    )

                    print(
                        f"   ✓ Sauvegardé dans {OUTPUT_FILE}"
                    )

            except Exception as e:

                print(
                    f"   ❌ Erreur agence : {e}"
                )

                # On continue avec l'agence suivante
                continue

            # ------------------------------------------------
            # Pause entre agences
            # ------------------------------------------------

            await random_wait(
                page,
                2,
                4
            )

        await browser.close()

    print("\n" + "=" * 70)

    print("SCRAPING TERMINÉ")

    print("=" * 70)

    print(
        f"Résultats : {OUTPUT_FILE}"
    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())