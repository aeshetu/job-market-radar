"""Turn a raw HN comment (HTML) into clean text plus structured posting fields.

HN hiring posts loosely follow the convention
    Company | Role(s) | Location | REMOTE / ONSITE | Salary | ...
on the first line, followed by free text. Everything here is rule-based and
deliberately transparent; tests/test_parse.py pins down the edge cases.
"""
from __future__ import annotations

import html
import re

_P_RE = re.compile(r"<p>|<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\w+")
_WS_RE = re.compile(r"[ \t ]+")


def html_to_text(raw: str | None) -> str:
    """HN comment HTML -> plain text, one paragraph per line."""
    if not raw:
        return ""
    s = _P_RE.sub("\n", raw)
    s = _TAG_RE.sub("", s)  # HN link text is the URL itself, so nothing is lost
    s = html.unescape(s)
    s = _WS_RE.sub(" ", s)
    return "\n".join(line.strip() for line in s.split("\n") if line.strip())


def strip_urls(text: str) -> str:
    """Remove URLs and emails so 'go.dev' or 'react@corp.com' can't fake a skill match."""
    return _URL_RE.sub(" ", text)


def header_of(text: str) -> str:
    return text.split("\n", 1)[0][:300]


# --------------------------------------------------------------------------- company
_SPLIT_RE = re.compile(r"\s*(?:\||•|·|\s[-–—]\s)\s*")


def company_of(header: str) -> str:
    name = _SPLIT_RE.split(header, maxsplit=1)[0]
    name = re.sub(r"\s*[\(\[].*?[\)\]]", "", name)  # "Acme (YC W21)" -> "Acme"
    return name.strip(" :,.")[:80]


def company_key(name: str) -> str:
    """Normalised key used to group reposts by the same company."""
    key = re.sub(r"[^a-z0-9]+", " ", name.lower())
    key = re.sub(r"\b(inc|llc|ltd|gmbh|corp|co|the|labs?|hq|ai|io)\b", " ", key)
    return " ".join(key.split())


# ------------------------------------------------------------------------- work mode
_NEG_REMOTE = re.compile(
    r"\b(?:no|not|non)[- ]remote\b"
    r"|\bremote\s*(?:[:=]|-)?\s*(?:no|not (?:possible|available|an option))\b"
    r"|\bno remote\b",
    re.IGNORECASE,
)
_REMOTE = re.compile(r"\bremote\b|\bwork from (?:home|anywhere)\b|\bWFH\b", re.IGNORECASE)
_HYBRID = re.compile(r"\bhybrid\b", re.IGNORECASE)
_ONSITE = re.compile(r"\bon[- ]?site\b|\bin[- ](?:office|person)\b", re.IGNORECASE)


def work_mode(header: str, body: str) -> str:
    """'remote' (remote allowed), 'hybrid', 'onsite', or 'unspecified'.

    The header is authoritative when it says anything; otherwise fall back to the body.
    """
    for text in (header, body):
        cleaned = _NEG_REMOTE.sub(" ", text)
        if _REMOTE.search(cleaned):
            return "remote"
        if _HYBRID.search(cleaned):
            return "hybrid"
        if _ONSITE.search(cleaned) or _NEG_REMOTE.search(text):
            return "onsite"
    return "unspecified"


# ------------------------------------------------------------------------------ visa
_VISA = re.compile(r"\bvisas?\b|\bH-?1B\b|\bsponsorship\b", re.IGNORECASE)
_NO_VISA = re.compile(
    r"\bno visas?\b|\b(?:cannot|can't|can not|unable to|do not|don't|not able to|won't|will not)\s+"
    r"(?:\w+\s+){0,2}sponsor|\bno (?:visa )?sponsorship\b|\bwithout (?:visa )?sponsorship\b"
    r"|\bsponsorship (?:is )?not\b|\bvisa sponsorship:? no\b",
    re.IGNORECASE,
)


def offers_visa(text: str) -> bool:
    return bool(_VISA.search(text)) and not _NO_VISA.search(text)


# ------------------------------------------------------------------------- seniority
SENIORITY = {
    "intern": r"\bintern(?:ship)?s?\b|\bco-?op\b",
    "junior": r"\bjunior\b|\bjr\.?(?=\s)|\bentry[- ]level\b|\bnew[- ]grads?\b|\brecent grad(?:uate)?s?\b",
    "senior": r"\bsenior\b|\bsr\.?(?=\s)",
    "staff_principal": r"\bstaff\s+(?:\w+\s+)?(?:engineer|scientist|developer|swe)s?\b|\bprincipal\b",
    "manager_exec": r"\b(?:engineering|eng) managers?\b|\bhead of\b|\bVP\b|\bdirector\b|\bCTO\b"
                    r"|\btech(?:nical)? leads?\b|\bteam leads?\b",
}
_SENIORITY_RE = {k: re.compile(v, re.IGNORECASE) for k, v in SENIORITY.items()}


def seniority_flags(text: str) -> dict[str, bool]:
    return {f"lvl_{k}": bool(r.search(text)) for k, r in _SENIORITY_RE.items()}


# ----------------------------------------------------------------------------- roles
# (?-i:...) keeps short acronyms case-sensitive so "ml" in prose can't match.
_ROLE_NOUN = r"(?:engineers?|scientists?|researchers?|developers?|leads?|engineering|research)"
ROLES = {
    "ml_ai": rf"\b(?:machine learning|(?-i:ML)|(?-i:AI)|deep learning|(?-i:LLM)|(?-i:NLP)|computer vision"
             rf"|(?-i:MLOps)|applied (?-i:AI)|(?-i:AI)/(?-i:ML)|(?-i:ML)/(?-i:AI))\b.{{0,20}}?\b{_ROLE_NOUN}\b"
             r"|\b(?:research|applied) scientists?\b|\b(?-i:MLE)s?\b",
    "data_science": r"\bdata scien(?:ce|tists?)\b|\bdata analysts?\b|\banalytics engineers?\b"
                    r"|\bquant(?:itative)? (?:analysts?|researchers?)\b|\bstatisticians?\b",
    "data_engineering": r"\bdata (?:engineers?|engineering|platform|infrastructure)\b|\b(?-i:ETL)\b",
    "backend": r"\bback[- ]?end\b",
    "frontend": r"\bfront[- ]?end\b|\b(?-i:UI) engineers?\b",
    "fullstack": r"\bfull[- ]?stack\b",
    "mobile": r"\b(?-i:iOS)\b|\bandroid\b|\bmobile (?:engineers?|developers?|apps?)\b",
    "devops_sre": r"\bdev[- ]?ops\b|\b(?-i:SRE)s?\b|\bsite reliability\b"
                  r"|\b(?:infrastructure|platform|cloud) engineers?\b",
    "security": r"\bsecurity engineers?\b|\bapp ?sec\b|\binfosec\b|\bsecurity (?:researchers?|analysts?)\b",
    "embedded_hardware": r"\bembedded\b|\bfirmware\b|\bhardware engineers?\b|\b(?-i:FPGA)\b"
                         r"|\belectrical engineers?\b|\brobotics engineers?\b",
    "product_design": r"\bproduct (?:managers?|designers?)\b|\b(?-i:UX)\b|\b(?:UI/UX|product|visual) designers?\b",
    "eng_management": r"\b(?:engineering|eng) managers?\b|\bhead of engineering\b"
                      r"|\b(?-i:VP),? (?:of )?engineering\b|\bdirector of engineering\b|\b(?-i:CTO)\b",
}
_ROLE_RE = {k: re.compile(v, re.IGNORECASE) for k, v in ROLES.items()}


def role_text(header: str, body: str) -> str:
    """Header plus short body lines: job titles are usually listed as short lines,
    while long lines are prose ("we use machine learning to...")."""
    short = [ln for ln in body.split("\n")[1:] if len(ln) <= 100]
    return "\n".join([header, *short])


def role_flags(text: str) -> dict[str, bool]:
    return {f"role_{k}": bool(r.search(text)) for k, r in _ROLE_RE.items()}


# ----------------------------------------------------------------------------- metro
METROS = {
    "SF Bay Area": r"\bSan Francisco\b|\b(?-i:SF)\b|\bBay Area\b|\bPalo Alto\b|\bMountain View\b"
                   r"|\bMenlo Park\b|\bSan Mateo\b|\bSan Jose\b|\bOakland\b|\bBerkeley\b|\bSunnyvale\b"
                   r"|\bRedwood City\b|\bCupertino\b|\bSanta Clara\b|\bEmeryville\b|\bBurlingame\b",
    "New York": r"\bNew York\b|\b(?-i:NYC)\b|\b(?-i:NY)\b|\bBrooklyn\b|\bManhattan\b",
    "Seattle": r"\bSeattle\b|\bBellevue\b|\bRedmond\b|\bKirkland\b",
    "Boston": r"\bBoston\b|\bCambridge,? (?-i:MA)\b|\bSomerville\b",
    "Los Angeles": r"\bLos Angeles\b|\b(?-i:LA)\b|\bSanta Monica\b|\bPasadena\b|\bCulver City\b|\bIrvine\b",
    "Austin": r"\bAustin\b",
    "Chicago": r"\bChicago\b",
    "Denver/Boulder": r"\bDenver\b|\bBoulder\b",
    "Washington DC": r"\bWashington,? (?-i:D\.?C)\b|\b(?-i:DC)\b",
    "London": r"\bLondon\b",
    "Berlin": r"\bBerlin\b",
    "Toronto": r"\bToronto\b",
    "Amsterdam": r"\bAmsterdam\b",
}
_METRO_RE = {k: re.compile(v, re.IGNORECASE) for k, v in METROS.items()}
US_METROS = {"SF Bay Area", "New York", "Seattle", "Boston", "Los Angeles", "Austin",
             "Chicago", "Denver/Boulder", "Washington DC"}


def metro_of(header: str) -> str:
    """Earliest metro mentioned in the header (posts list the HQ first)."""
    best, best_pos = "Other/unspecified", len(header) + 1
    for name, rx in _METRO_RE.items():
        m = rx.search(header)
        if m and m.start() < best_pos:
            best, best_pos = name, m.start()
    return best
