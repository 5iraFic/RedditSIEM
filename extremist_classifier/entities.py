"""
Entity lists for Islamic extremist content detection.

FRIENDLY entities: groups, figures, and concepts that jihadist-extremists
    sympathize with, glorify, or identify with.
ENEMY entities: groups and concepts that jihadist-extremists target or
    express hatred toward.

Each entry maps a canonical name to a list of regex-ready surface forms
(case-insensitive matching with word boundaries applied at runtime).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


class EntityType(Enum):
    FRIENDLY = "friendly"
    ENEMY = "enemy"


# ---------------------------------------------------------------------------
# Entity dictionaries
# ---------------------------------------------------------------------------

# Surface forms are matched with re.IGNORECASE + \b word boundaries.
# Order inside each list matters when two patterns overlap: put longer /
# more specific forms first.

FRIENDLY_ENTITIES: dict[str, list[str]] = {
    "Islamic State": [
        r"islamic\s+state",
        r"isis",
        r"isil",
        r"daesh",
        r"is\s+caliphate",
        r"dawlah\s+islamiyyah",
        r"dawla",
        r"wilayat",
    ],
    "Al-Qaeda": [
        r"al[\s\-]?qaeda",
        r"al[\s\-]?qa'ida",
        r"alqaeda",
        r"base\s+of\s+jihad",     # literal translation
    ],
    "Mujahideen": [
        r"mujahid(?:een|in|un)?",
        r"mujahed(?:een|in)?",
        r"fighters?\s+(?:of|for)\s+allah",
        r"soldiers?\s+of\s+allah",
        r"soldiers?\s+of\s+the\s+caliphate",
        r"ghazi",
    ],
    "Martyrs": [
        r"shaheed",
        r"shahid",
        r"shuhada",
        r"istishhad",
        r"istishad",
        r"martyrdom\s+operation",
        r"martyrdom",
        r"martyr(?:s)?",
    ],
    "Jihad": [
        r"jihad(?:ist(?:s)?)?",
        r"holy\s+war",
        r"qital",
        r"ghazwa",
        r"ghazwah",
        r"ribat",
    ],
    "Caliphate": [
        r"khilaf(?:ah|a)",
        r"khalif(?:ah|a)?",
        r"caliphate",
        r"caliph",
    ],
    "Taliban": [
        r"taliban",
        r"talib(?:an)?",
        r"islamic\s+emirate\s+of\s+afghanistan",
        r"iea",
    ],
    "Hamas": [
        r"hamas",
        r"izz\s+ad[\s\-]din\s+al[\s\-]qassam",
        r"qassam\s+brigades?",
    ],
    "HTS / Jabhat al-Nusra": [
        r"hayat\s+tahrir\s+al[\s\-]sham",
        r"\bhts\b",
        r"jabhat\s+al[\s\-]nusra",
        r"jabhat\s+fath\s+al[\s\-]sham",
        r"nusra\s+front",
    ],
    "Al-Shabaab": [
        r"al[\s\-]?shabaab",
        r"harakat\s+al[\s\-]shabaab",
    ],
    "Boko Haram / ISWAP": [
        r"boko\s+haram",
        r"iswap",
        r"jamaat\s+ahl\s+as[\s\-]sunnah",
    ],
    "Hezbollah": [
        r"hezb(?:o|u)?llah",
        r"hizb[\s\-]?ullah",
        r"party\s+of\s+god",        # in glorifying context
    ],
    "Osama bin Laden": [
        r"osama\s+bin\s+laden",
        r"usama\s+bin\s+laden",
        r"bin\s+laden",
        r"sheikh\s+osama",
    ],
    "Baghdadi": [
        r"abu\s+bakr\s+al[\s\-]baghdadi",
        r"al[\s\-]baghdadi",
        r"caliph\s+ibrahim",
    ],
    "Zawahiri": [
        r"(?:ayman\s+)?al[\s\-]zawahiri",
        r"zawahiri",
    ],
    "Anwar al-Awlaki": [
        r"anwar\s+al[\s\-]awlaki",
        r"al[\s\-]awlaki",
        r"awlaki",
    ],
    "Ummah / Brotherhood": [
        r"\bummah\b",
        r"muslim\s+brotherhood",
        r"ikhwan\s+(?:al[\s\-])?muslimeen",
        r"brothers?\s+in\s+(?:islam|faith)",
        r"ansar\s+al",              # prefix used by many affiliated groups
    ],
    "Bay'ah / Pledge": [
        r"bay['']?ah",
        r"bay['']?at",
        r"pledge\s+of\s+allegiance\s+to\s+(?:isis|is|caliph)",
        r"hijra\s+to\s+(?:isis|is|the\s+caliphate)",
    ],
}

ENEMY_ENTITIES: dict[str, list[str]] = {
    "Americans / USA": [
        r"american(?:s)?",
        r"\busa\b",
        r"\bu\.s\.a\.\b",
        r"\bu\.s\.\b",
        r"\bus\s+(?:government|military|army|forces|troops|soldiers?)",
        r"uncle\s+sam",
        r"yankee(?:s)?",
    ],
    "Jews / Israel / Zionists": [
        r"jew(?:s|ish)?",
        r"israel(?:i(?:s)?)?",
        r"zionist(?:s)?",
        r"zionism",
        r"\bidf\b",
        r"mossad",
        r"rothschild",              # antisemitic conspiracy term
        r"yahud(?:i)?",
    ],
    "Kuffar / Infidels": [
        r"kuffar",
        r"kafir(?:s|oon|een)?",
        r"kufr",
        r"infidel(?:s)?",
        r"unbeliever(?:s)?",
        r"disbeliever(?:s)?",
        r"non[\s\-]?believer(?:s)?",
    ],
    "Crusaders": [
        r"crusader(?:s)?",
        r"crusade(?:s)?",
        r"salibi",
    ],
    "Christians": [
        # Only flag in explicitly anti-Christian extremist context;
        # ABSA sentiment toward this entity disambiguates
        r"christian(?:s|ity)?",
        r"nasrani",
        r"church\s+of",
    ],
    "Apostates / Murtad": [
        r"apostate(?:s)?",
        r"apostasy",
        r"murtad(?:d)?",
        r"riddah",
    ],
    "Shia / Rafidah": [
        r"shia(?:ts?)?",
        r"shi['']a(?:ts?)?",
        r"shiite(?:s)?",
        r"rafidah",
        r"rawafid",
        r"safavid(?:s)?",
        r"twelver(?:s)?",
    ],
    "Western Governments / NATO": [
        r"\bnato\b",
        r"pentagon",
        r"\bcia\b",
        r"\bfbi\b",
        r"white\s+house",
        r"european\s+union",
        r"\beu\b",
        r"western\s+(?:government|alliance|forces?|coalition)",
    ],
    "Taghut / Tyrannical Rulers": [
        r"taghut",
        r"tawagheet",
        r"tyrant\s+(?:ruler|government|regime)",
        r"apostate\s+(?:ruler|government|regime)",
        r"secular\s+(?:government|regime)",
    ],
    "Mushrikeen / Polytheists": [
        r"mushrik(?:een|in|un)?",
        r"polytheist(?:s)?",
        r"idolater(?:s)?",
        r"pagan(?:s)?",
        r"mushrik",
    ],
    "Russia": [
        # Frequently an IS enemy entity in Caucasus / Syria contexts
        r"russian(?:s)?",
        r"\brussia\b",
        r"putin",
        r"\bfsb\b",
    ],
    "Assad / Syrian Regime": [
        r"bashar\s+al[\s\-]?assad",
        r"al[\s\-]?assad",
        r"syrian\s+(?:army|regime|government|forces?)",
        r"alawite(?:s)?",
        r"nusayri(?:s)?",
    ],
}


# ---------------------------------------------------------------------------
# Compiled pattern cache
# ---------------------------------------------------------------------------

def _compile(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(r"\b" + p + r"\b", re.IGNORECASE) for p in patterns]


_FRIENDLY_COMPILED: dict[str, list[re.Pattern[str]]] = {
    canonical: _compile(patterns)
    for canonical, patterns in FRIENDLY_ENTITIES.items()
}

_ENEMY_COMPILED: dict[str, list[re.Pattern[str]]] = {
    canonical: _compile(patterns)
    for canonical, patterns in ENEMY_ENTITIES.items()
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EntityMatch:
    canonical: str
    matched_text: str
    entity_type: EntityType
    sentence: str
    char_start: int   # start offset inside `sentence`
    char_end: int     # end offset inside `sentence`


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def iter_entity_matches(
    sentence: str,
    include_friendly: bool = True,
    include_enemy: bool = True,
) -> Iterator[EntityMatch]:
    """
    Yield all EntityMatch objects found in *sentence*.

    One match is yielded per (entity_type, canonical_name) per sentence,
    using the first pattern that fires.  This avoids emitting duplicate
    evidence for the same canonical entity.
    """
    seen: set[tuple[EntityType, str]] = set()

    def _scan(
        compiled: dict[str, list[re.Pattern[str]]],
        etype: EntityType,
    ) -> Iterator[EntityMatch]:
        for canonical, patterns in compiled.items():
            key = (etype, canonical)
            if key in seen:
                continue
            for pat in patterns:
                m = pat.search(sentence)
                if m:
                    seen.add(key)
                    yield EntityMatch(
                        canonical=canonical,
                        matched_text=m.group(0),
                        entity_type=etype,
                        sentence=sentence,
                        char_start=m.start(),
                        char_end=m.end(),
                    )
                    break  # one match per canonical is enough

    if include_friendly:
        yield from _scan(_FRIENDLY_COMPILED, EntityType.FRIENDLY)
    if include_enemy:
        yield from _scan(_ENEMY_COMPILED, EntityType.ENEMY)


def has_any_entity(text: str) -> bool:
    """Quick check: does the text contain at least one tracked entity?"""
    for sent in [text]:   # caller may pass a full doc; sentence split happens upstream
        for _ in iter_entity_matches(sent):
            return True
    return False
