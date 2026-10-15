"""Tests for wikipedia_source's pure helpers (no network calls)."""
from __future__ import annotations

from matchmind.knowledge_base.wikipedia_source import _parse_skill_links, article_url


_SAMPLE_WIKITEXT = """
The following is a list of individual skills in [[association football]].

== Kicking ==
{{Main article|Kick (association football)}}
Push kick, instep kick, [[Chip (association football)|chip]].

== Dribbling ==
[[Cruyff turn]], [[Step over]].

== See also ==
* [[Association football positions]]
* [[Association football tactics]]

{{Association football}}
"""


def test_parse_skill_links_excludes_see_also_and_navbox():
    links = _parse_skill_links(_SAMPLE_WIKITEXT)
    assert "Chip (association football)" in links
    assert "Cruyff turn" in links
    assert "Step over" in links
    # "See also" section links must be excluded
    assert "Association football positions" not in links
    assert "Association football tactics" not in links


def test_parse_skill_links_excludes_self_link_and_dedupes():
    links = _parse_skill_links(_SAMPLE_WIKITEXT)
    assert "association football" not in [t.lower() for t in links]
    assert len(links) == len(set(links))


def test_article_url():
    assert article_url("Half space") == "https://en.wikipedia.org/wiki/Half_space"
