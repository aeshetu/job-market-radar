"""Extract advertised annual salary ranges from free-text job posts.

Handles "$120k-$160k", "$120,000 – 160,000", "€80.000 - 95.000", "120-150k USD",
"$200-300K", "£70k+". Rejects hourly/monthly rates, funding amounts ("$10-20M"),
and anything outside a plausible annual range. Precision over recall: a missed
salary costs one row, while a wrong one biases the salary model.
"""
from __future__ import annotations

import re
import statistics

_NUM = r"(?:\d{1,3}(?:[,.]\d{3})+|\d{1,7})(?:\.\d{1,2})?(?![\d,]\d)"
_CODES = r"USD|EUR|GBP|CAD|AUD|CHF"


def _prefix(name: str) -> str:
    return rf"(?P<{name}>US\$|CA?\$|C\$|AU?\$|\$|€|£|(?:{_CODES})\s?)"


_RANGE_RE = re.compile(
    rf"{_prefix('cur1')}?\s?(?P<lo>{_NUM})\s?(?P<lok>[kK](?![a-zA-Z]))?"
    rf"\s*(?:-|–|—|‑|to)\s*"
    rf"{_prefix('cur2')}?\s?(?P<hi>{_NUM})\s?(?P<hik>[kK](?![a-zA-Z]))?"
    rf"(?:\s?(?P<cur3>{_CODES}|€|£)\b)?"
)
_SINGLE_RE = re.compile(
    rf"{_prefix('cur1')}\s?(?P<v>{_NUM})\s?(?P<vk>[kK](?![a-zA-Z]))?\+?"
)
_NOT_ANNUAL_AFTER = re.compile(
    r"^\s*(?:/|per|an?|each)?\s*(?:hr|hour|h|hourly|month|mo|week|wk|day|daily)\b"
    r"|^\s*(?:M|MM|B|bn|mn|million|billion)\b"
    r"|^\s*(?:in |of )?(?:(?:signing|sign-on|sign on|joining|relocation|referral|annual|performance|"
    r"education|learning|equipment|home office)\s)?"
    r"(?:funding|raised|revenue|ARR|MRR|GMV|seed|series|valuation|bonus|stipend|grant|budget|allowance)"
    r"|^\s*(?:or less|or more|in (?:revenue|sales|funding))\b",
    re.IGNORECASE,
)
_NOT_ANNUAL_BEFORE = re.compile(
    r"(?:raised|raise|funding|funded|revenue|ARR|MRR|valuation|bonus|stipend|budget|grant|managing|"
    r"invested|investment|making|made|saved|savings)\s*\S*\s*$",
    re.IGNORECASE,
)

_CURRENCY = {
    "$": "USD", "US$": "USD", "USD": "USD",
    "CA$": "CAD", "C$": "CAD", "CAD": "CAD",
    "A$": "AUD", "AU$": "AUD", "AUD": "AUD",
    "€": "EUR", "EUR": "EUR", "£": "GBP", "GBP": "GBP", "CHF": "CHF",
}
MIN_ANNUAL, MAX_ANNUAL = 15_000, 1_500_000


def _to_float(num: str) -> float:
    m = re.fullmatch(r"(\d{1,3}(?:[,.]\d{3})+|\d{1,7})(?:\.(\d{1,2}))?", num)
    whole = re.sub(r"[,.]", "", m.group(1))
    return float(f"{whole}.{m.group(2) or 0}")


def _currency(*tokens: str | None) -> str | None:
    for t in tokens:
        if t:
            return _CURRENCY.get(t.strip())
    return None


def _context_ok(text: str, start: int, end: int) -> bool:
    return not (_NOT_ANNUAL_AFTER.search(text[end:end + 25])
                or _NOT_ANNUAL_BEFORE.search(text[max(0, start - 25):start]))


def find_salaries(text: str) -> list[dict]:
    """All plausible annual salary mentions: [{'low', 'high', 'currency'}]."""
    found, taken = [], []
    for m in _RANGE_RE.finditer(text):
        cur = _currency(m["cur3"], m["cur1"], m["cur2"])  # "$45k-55k CAD" is CAD
        if cur is None or not _context_ok(text, m.start(), m.end()):
            continue
        lo, hi = _to_float(m["lo"]), _to_float(m["hi"])
        if m["lok"]:
            lo *= 1000
        if m["hik"]:
            hi *= 1000
        if m["hik"] and not m["lok"] and lo < 1000:  # "$120-150k"
            lo *= 1000
        if m["lok"] and not m["hik"] and hi < 1000:  # "$120k-150"
            hi *= 1000
        if MIN_ANNUAL <= lo <= hi <= MAX_ANNUAL and hi / lo <= 4:
            found.append({"low": lo, "high": hi, "currency": cur})
            taken.append((m.start(), m.end()))

    for m in _SINGLE_RE.finditer(text):
        if any(s <= m.start() < e for s, e in taken):
            continue
        if not _context_ok(text, m.start(), m.end()):
            continue
        v = _to_float(m["v"]) * (1000 if m["vk"] else 1)
        if MIN_ANNUAL <= v <= MAX_ANNUAL:
            found.append({"low": v, "high": v, "currency": _currency(m["cur1"])})
    return found


def extract_salary(text: str) -> dict:
    """Summarise a post's salary mentions in its dominant currency."""
    hits = find_salaries(text)
    empty = {"salary_low": None, "salary_high": None, "salary_mid": None,
             "salary_currency": None, "salary_mentions": 0}
    if not hits:
        return empty
    currency = statistics.mode(h["currency"] for h in hits)
    hits = [h for h in hits if h["currency"] == currency]
    return {
        "salary_low": min(h["low"] for h in hits),
        "salary_high": max(h["high"] for h in hits),
        "salary_mid": statistics.median((h["low"] + h["high"]) / 2 for h in hits),
        "salary_currency": currency,
        "salary_mentions": len(hits),
    }
