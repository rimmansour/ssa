"""
Scraper d'avis Google Maps pour une liste d'agences Société Générale.

Entrée  : agences.csv  (séparateur ';', colonne obligatoire 'adresse',
          colonnes optionnelles 'id' et 'agence')
Sorties : avis_agences.jsonl (1 ligne par agence, reprise possible)
          avis_agences.csv   (1 ligne par avis, à plat, avec coordonnées GPS)
          debug_<id>.png     (screenshot de chaque fiche après le scroll)

Installation :
    pip install playwright
    playwright install chromium

Lancement :
    python scrape_avis_sg.py

NB : supprime avis_agences.jsonl avant de relancer si tu veux retraiter
les agences déjà scrapées.
"""

import asyncio
import csv
import json
import random
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from playwright.async_api import async_playwright

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
INPUT_CSV = "agences.csv"
OUTPUT_JSONL = "avis_agences.jsonl"
OUTPUT_CSV = "avis_agences.csv"

BANK_NAME = "Société Générale"
MAX_REVIEWS = 200          # max d'avis par agence (None = tout)
HEADLESS = False           # False conseillé : moins de blocages, debug plus simple
DELAY = (1.5, 3.5)         # pause aléatoire entre actions (secondes)
BETWEEN_PLACES = (4, 9)    # pause entre deux agences
MAX_STALL = 6              # nb de scrolls sans nouvel avis avant d'arrêter

# Sélecteurs Google Maps (changent de temps en temps : à ajuster si besoin)
SEL_PLACE_TITLE = "h1.DUwDvf"
SEL_RESULT_LINK = "a.hfpxzc"
SEL_REVIEW_CARD = "div.jftiEf"
SEL_MORE_BTN = "button.w8nwRe"

# Trouve le conteneur scrollable à partir d'une carte d'avis et le fait défiler
# (évite de dépendre des noms de classes du panneau)
FIND_AND_SCROLL_JS = """
() => {
  const card = document.querySelector('div.jftiEf');
  if (!card) return false;
  let el = card.parentElement;
  while (el && el !== document.body) {
    const s = getComputedStyle(el);
    if ((s.overflowY === 'auto' || s.overflowY === 'scroll') && el.scrollHeight > el.clientHeight) {
      el.scrollTo(0, el.scrollHeight);
      return true;
    }
    el = el.parentElement;
  }
  return false;
}
"""


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
async def pause(a=DELAY[0], b=DELAY[1]):
    await asyncio.sleep(random.uniform(a, b))


async def accept_cookies(page):
    """Gère la page de consentement Google (consent.google.com ou bandeau)."""
    candidates = [
        'button:has-text("Tout accepter")',
        'button:has-text("Accept all")',
        'button[aria-label="Tout accepter"]',
    ]
    for sel in candidates:
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=2500):
                await btn.click()
                await page.wait_for_load_state("domcontentloaded")
                await pause(1, 2)
                return
        except Exception:
            continue


async def dismiss_signin_popup(page):
    """Ferme la fenêtre 'Connectez-vous pour profiter pleinement de Google Maps'."""
    candidates = [
        page.get_by_role("button", name="Ignorer"),
        page.get_by_text("Ignorer", exact=True),
        page.get_by_role("button", name="Not now"),
    ]
    for loc in candidates:
        try:
            btn = loc.first
            if await btn.is_visible(timeout=1500):
                await btn.click()
                await pause(0.5, 1)
                return
        except Exception:
            continue


def load_addresses(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    for i, r in enumerate(rows):
        r["id"] = (r.get("id") or str(i + 1)).strip()
        r["adresse"] = r["adresse"].strip()
    return rows


def already_done(path):
    done = set()
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


def extract_gps(url):
    """Extrait (lat, lng) de l'URL d'une fiche Google Maps."""
    m = re.search(r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)", url)
    if not m:
        m = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+)", url)  # repli : centre de la carte
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


# --------------------------------------------------------------------------- #
# Étapes de scraping
# --------------------------------------------------------------------------- #
async def find_place(page, address):
    """Recherche '<banque> <adresse>' et ouvre la fiche du 1er résultat."""
    query = f"{BANK_NAME} {address}"
    url = f"https://www.google.com/maps/search/{quote(query)}?hl=fr"
    await page.goto(url, wait_until="domcontentloaded")
    await accept_cookies(page)
    await dismiss_signin_popup(page)

    try:
        await page.wait_for_selector(f"{SEL_PLACE_TITLE}, {SEL_RESULT_LINK}", timeout=15000)
    except Exception:
        return False

    # La popup peut aussi apparaître après le chargement
    await dismiss_signin_popup(page)

    # Si Maps affiche une liste de résultats, on clique sur le premier
    if not await page.locator(SEL_PLACE_TITLE).count():
        await page.locator(SEL_RESULT_LINK).first.click()
        await page.wait_for_selector(SEL_PLACE_TITLE, timeout=15000)
    await pause()
    return True


async def get_place_info(page):
    info = await page.evaluate(
        """() => {
            const t = s => (document.querySelector(s)?.innerText || '').trim();
            const addrBtn = document.querySelector('button[data-item-id="address"]');
            const ratingBlock = document.querySelector('div.F7nice');
            const countEl = ratingBlock?.querySelector('span[aria-label*="avis"]');
            return {
                name: t('h1.DUwDvf'),
                address: (addrBtn?.getAttribute('aria-label') || '').replace(/^Adresse\\s*:\\s*/i, ''),
                rating: ratingBlock?.querySelector('span[aria-hidden="true"]')?.innerText || '',
                total_reviews: (countEl?.getAttribute('aria-label') || '').replace(/\\D/g, ''),
            };
        }"""
    )
    norm = lambda s: s.lower().replace("é", "e").replace("è", "e")
    info["is_bank_match"] = bool(re.search(r"\b(societe generale|sg)\b", norm(info["name"])))

    # L'URL met parfois un instant à se mettre à jour avec les coordonnées
    for _ in range(5):
        if "!3d" in page.url:
            break
        await asyncio.sleep(0.5)
    info["url"] = page.url
    info["lat"], info["lng"] = extract_gps(page.url)
    return info


async def open_reviews_tab(page):
    await dismiss_signin_popup(page)
    for sel in [
        'button[role="tab"]:has-text("Avis")',
        'button[aria-label*="Avis"]',
        'div.F7nice button',
    ]:
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=3000):
                await btn.click()
                # Vérifie que l'onglet Avis est réellement sélectionné
                await page.wait_for_selector(
                    'button[role="tab"][aria-selected="true"]:has-text("Avis")',
                    timeout=10000,
                )
                await page.wait_for_selector(SEL_REVIEW_CARD, timeout=10000)
                await pause()
                return True
        except Exception:
            continue
    return False


async def sort_by_newest(page):
    """Trie les avis par 'Les plus récents'. Renvoie True si le tri a été appliqué."""
    await dismiss_signin_popup(page)

    sort_buttons = [
        page.get_by_role("button", name=re.compile(r"trier|sort", re.I)),
        page.locator('button[aria-label*="Trier"], button[data-value="Trier"], button[data-value="Sort"]'),
    ]
    for sort_btn in sort_buttons:
        try:
            await sort_btn.first.click(timeout=5000)
            await pause(0.8, 1.5)
            option = page.get_by_role("menuitemradio", name=re.compile(r"récent|newest", re.I)).first
            await option.click(timeout=5000)
            await pause(2.0, 3.0)  # laisse la liste se recharger
            return True
        except Exception:
            continue

    print("    ! tri par 'plus récents' impossible (bouton ou option introuvable)")
    return False


async def scroll_reviews(page, max_reviews):
    """Fait défiler le panneau d'avis jusqu'à épuisement ou limite atteinte."""
    last, stall = 0, 0
    while stall < MAX_STALL:
        await dismiss_signin_popup(page)
        count = await page.locator(SEL_REVIEW_CARD).count()
        if max_reviews and count >= max_reviews:
            break
        stall = stall + 1 if count == last else 0
        last = count
        scrolled = await page.evaluate(FIND_AND_SCROLL_JS)
        if not scrolled:
            await page.mouse.wheel(0, 3000)  # repli
        await pause(1.0, 2.0)

        # En affichage limité, Maps s'arrête à 5 avis et propose "Voir plus d'avis (N)"
        try:
            more = page.get_by_text(re.compile(r"Voir plus d.avis", re.I)).first
            if await more.is_visible(timeout=800):
                await more.scroll_into_view_if_needed()
                await more.click()
                await pause(2.0, 3.0)
                await dismiss_signin_popup(page)
        except Exception:
            pass
        await pause(1.0, 2.0)


async def expand_and_extract(page):
    # Déplie les commentaires tronqués ("Plus")
    await page.evaluate(
        f"() => document.querySelectorAll('{SEL_MORE_BTN}').forEach(b => b.click())"
    )
    await pause(0.8, 1.5)
    return await page.evaluate(
        f"""() => [...document.querySelectorAll('{SEL_REVIEW_CARD}')].map(c => {{
            const q = s => c.querySelector(s);
            const txt = s => (q(s)?.innerText || '').trim();
            const stars = q('span.kvMYJc')?.getAttribute('aria-label') || '';
            return {{
                review_id: c.getAttribute('data-review-id'),
                author: txt('div.d4r55'),
                author_info: txt('div.RfnDt'),
                rating: (stars.match(/(\\d)/) || [])[1] || null,
                date: txt('span.rsqaWe'),
                text: txt('div.MyEned span.wiI7pd'),
                owner_response: txt('div.CDe7pd div.wiI7pd'),
                owner_response_date: txt('div.CDe7pd span.DZSIDd'),
            }};
        }})"""
    )


async def scrape_agency(context, row):
    page = await context.new_page()
    result = {
        "id": row["id"],
        "adresse_input": row["adresse"],
        "status": "ok",
        "scraped_at": datetime.now().isoformat(timespec="seconds"),
        "place": {},
        "reviews": [],
    }
    try:
        if not await find_place(page, row["adresse"]):
            result["status"] = "place_not_found"
            return result

        result["place"] = await get_place_info(page)

        if not await open_reviews_tab(page):
            result["status"] = "no_reviews_tab"
            return result

        await sort_by_newest(page)
        await scroll_reviews(page, MAX_REVIEWS)
        await page.screenshot(path=f"debug_{row['id']}.png")
        reviews = await expand_and_extract(page)

        # Dédoublonnage + limite
        seen, unique = set(), []
        for r in reviews:
            key = r["review_id"] or (r["author"], r["date"], r["text"][:40])
            if key not in seen:
                seen.add(key)
                unique.append(r)
        result["reviews"] = unique[:MAX_REVIEWS] if MAX_REVIEWS else unique

        if not result["place"].get("is_bank_match"):
            result["status"] = "ok_but_name_mismatch"  # à vérifier manuellement
    except Exception as e:
        result["status"] = f"error: {type(e).__name__}: {e}"
    finally:
        await page.close()
    return result


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_flat_csv(jsonl_path, csv_path):
    cols = [
        "agence_id", "adresse_input", "status", "place_name", "place_address",
        "place_rating", "place_total_reviews", "place_url", "lat", "lng",
        "review_id", "author", "author_info", "rating", "date", "text",
        "owner_response", "owner_response_date", "scraped_at",
    ]
    with open(jsonl_path, encoding="utf-8") as fin, \
         open(csv_path, "w", newline="", encoding="utf-8-sig") as fout:
        w = csv.DictWriter(fout, fieldnames=cols, delimiter=";")
        w.writeheader()
        for line in fin:
            a = json.loads(line)
            p = a.get("place", {})
            base = {
                "agence_id": a["id"], "adresse_input": a["adresse_input"],
                "status": a["status"], "place_name": p.get("name"),
                "place_address": p.get("address"), "place_rating": p.get("rating"),
                "place_total_reviews": p.get("total_reviews"), "place_url": p.get("url"),
                "lat": p.get("lat"), "lng": p.get("lng"),
                "scraped_at": a["scraped_at"],
            }
            if not a["reviews"]:
                w.writerow(base)
            for r in a["reviews"]:
                w.writerow({**base, **{k: r.get(k) for k in (
                    "review_id", "author", "author_info", "rating", "date",
                    "text", "owner_response", "owner_response_date")}})


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
async def main():
    rows = load_addresses(INPUT_CSV)
    done = already_done(OUTPUT_JSONL)
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} agences au total, {len(todo)} à traiter.")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=HEADLESS,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            locale="fr-FR",
            timezone_id="Europe/Paris",
            viewport={"width": 1366, "height": 850},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
        )

        for i, row in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {row['adresse']}")
            res = await scrape_agency(context, row)
            print(f"    -> {res['status']} | {len(res['reviews'])} avis | {res['place'].get('name')}")
            with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            await pause(*BETWEEN_PLACES)

        await browser.close()

    export_flat_csv(OUTPUT_JSONL, OUTPUT_CSV)
    print(f"Terminé. Export : {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())