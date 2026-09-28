"""
Wikipedia Source — discovery + fetching for the KB ingestion pipeline.

Per knowledge_bonus.md: Wikipedia's football-tactics coverage is structured
into ready-made categories and is CC BY-SA licensed (reuse/paraphrase allowed
with attribution) — a much better fit than crawling arbitrary copyrighted
sites. Three categories are used:

    Category:Association football tactics       (~27 articles)
    Category:Association football terminology    (~211 articles — noisy, the
                                                    LLM synthesis step filters
                                                    out non-tactical entries)
    Category:Football skills                     (near-empty as a category;
                                                    the actual list of ~32
                                                    individual skills lives as
                                                    links inside the article
                                                    "List of association
                                                    football skills")

Uses the public MediaWiki API directly (JSON, no scraping/HTML parsing) via
httpx, which is already a project dependency.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_ARTICLE_BASE = "https://en.wikipedia.org/wiki/"

# Identify ourselves per Wikipedia's API etiquette (https://meta.wikimedia.org/
# wiki/User-Agent_policy) without embedding any personal contact info.
_USER_AGENT = "MatchMind-Tactical-KB-Builder/1.0 (non-commercial educational research project)"

# Titles that are structural/meta pages, not real tactical concepts — skip
# before spending an LLM call on them.
_SKIP_PREFIXES = ("List of", "Wikipedia:", "Template:", "Category:", "Portal:")

DEFAULT_CATEGORIES = {
    "Association football tactics": "attacking",
    "Association football terminology": "terminology",
}
SKILLS_LIST_ARTICLE = "List of association football skills"


@dataclass
class WikiArticle:
    title: str
    category_hint: str  # a best-guess TacticalConcept.category for this bucket


def _client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": _USER_AGENT}, timeout=15.0)


def fetch_category_members(
    category: str,
    *,
    limit: int = 500,
    client: httpx.Client | None = None,
) -> list[str]:
    """All page titles (namespace 0 = articles) directly in a Wikipedia category."""
    own_client = client is None
    client = client or _client()
    titles: list[str] = []
    cmcontinue: str | None = None
    try:
        while len(titles) < limit:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmtype": "page",
                "cmlimit": min(500, limit - len(titles)),
                "format": "json",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            resp = client.get(WIKI_API, params=params)
            resp.raise_for_status()
            data = resp.json()
            members = data.get("query", {}).get("categorymembers", [])
            titles.extend(
                m["title"] for m in members if not m["title"].startswith(_SKIP_PREFIXES)
            )
            cmcontinue = data.get("continue", {}).get("cmcontinue")
            if not cmcontinue:
                break
        return titles[:limit]
    finally:
        if own_client:
            client.close()


def fetch_list_article_links(
    page_title: str,
    *,
    limit: int = 40,
    client: httpx.Client | None = None,
) -> list[str]:
    """Individual-skill article titles linked from a Wikipedia list page.

    Used for the "skills" bucket: Category:Football skills itself is nearly
    empty, but "List of association football skills" enumerates them under
    headings (Kicking, Dribbling, Passing, ...) each linking to one article
    per skill (nutmeg, step-over, ...).

    Deliberately parses the raw wikitext rather than calling prop=links: that
    API returns every link on the *rendered* page, including ones pulled in
    by the "See also" section and the navbox templates transcluded at the
    bottom (e.g. {{Association football}}) — which drags in unrelated pages
    like "12th man" or "Alcohol in association football". Wikitext lets us
    cut the page off at "== See also ==" and only look at the skill sections
    above it.
    """
    own_client = client is None
    client = client or _client()
    try:
        resp = client.get(
            WIKI_API,
            params={
                "action": "parse",
                "page": page_title,
                "prop": "wikitext",
                "format": "json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
        if not wikitext:
            logger.warning(f"No wikitext for {page_title!r}")
            return []
        return _parse_skill_links(wikitext)[:limit]
    finally:
        if own_client:
            client.close()


def _parse_skill_links(wikitext: str) -> list[str]:
    """Pure helper: wikilink targets in the body (before '== See also =='),
    excluding the generic 'association football' self-link. Split out from
    fetch_list_article_links so it's unit-testable without a network call.
    """
    body = wikitext.split("== See also ==")[0]
    raw_links = re.findall(r"\[\[([^\]|#]+)(?:\|[^\]]+)?\]\]", body)

    titles: list[str] = []
    seen: set[str] = set()
    for raw in raw_links:
        t = raw.replace("_", " ").strip()
        if t and t.lower() != "association football" and not t.startswith(_SKIP_PREFIXES) and t not in seen:
            seen.add(t)
            titles.append(t)
    return titles


def fetch_extract(
    title: str,
    *,
    max_chars: int = 4000,
    client: httpx.Client | None = None,
) -> str | None:
    """Plain-text extract (intro + body, HTML/markup stripped) for one article."""
    own_client = client is None
    client = client or _client()
    try:
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "redirects": 1,
            "format": "json",
        }
        resp = client.get(WIKI_API, params=params)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if "missing" in page:
                return None
            extract = page.get("extract", "")
            if extract:
                return extract[:max_chars]
        return None
    except httpx.HTTPError as exc:
        logger.warning(f"Failed to fetch extract for {title!r}: {exc}")
        return None
    finally:
        if own_client:
            client.close()


def article_url(title: str) -> str:
    return WIKI_ARTICLE_BASE + title.replace(" ", "_")


def discover_articles(
    *,
    max_per_category: int = 40,
    max_skills: int = 35,
) -> list[WikiArticle]:
    """Full discovery step: tactics + terminology categories, plus the skills list."""
    articles: list[WikiArticle] = []
    with _client() as client:
        for category, category_hint in DEFAULT_CATEGORIES.items():
            titles = fetch_category_members(category, limit=max_per_category, client=client)
            logger.info(f"Category {category!r}: {len(titles)} articles")
            articles.extend(WikiArticle(title=t, category_hint=category_hint) for t in titles)
            time.sleep(0.2)

        skill_titles = fetch_list_article_links(SKILLS_LIST_ARTICLE, limit=max_skills, client=client)
        logger.info(f"Skills list {SKILLS_LIST_ARTICLE!r}: {len(skill_titles)} linked articles")
        articles.extend(WikiArticle(title=t, category_hint="individual") for t in skill_titles)

    # De-duplicate titles that showed up in more than one bucket, keep first.
    seen: set[str] = set()
    unique: list[WikiArticle] = []
    for a in articles:
        if a.title not in seen:
            seen.add(a.title)
            unique.append(a)
    return unique
