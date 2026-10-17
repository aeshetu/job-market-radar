import pytest

from radar.parse import (company_of, html_to_text, metro_of, offers_visa, role_flags,
                         role_text, seniority_flags, strip_urls, work_mode)


def test_html_to_text():
    raw = ('Acme | Backend &amp; ML | SF<p>We&#x27;re hiring! See '
           '<a href="https:&#x2F;&#x2F;acme.com">https:&#x2F;&#x2F;acme.com</a><p>Thanks')
    assert html_to_text(raw) == "Acme | Backend & ML | SF\nWe're hiring! See https://acme.com\nThanks"
    assert html_to_text(None) == ""


def test_strip_urls():
    assert "go.dev" not in strip_urls("apply at https://go.dev/jobs or jobs@react.io")


@pytest.mark.parametrize("header, expected", [
    ("Acme (YC W21) | Senior Engineer | Remote", "Acme"),
    ("Shepherd (Series B) | ONSITE | San Francisco", "Shepherd"),
    ("PrairieLearn (Remote US) — Full-Stack Engineer", "PrairieLearn"),
    ("Stripe - Backend Engineer - SF", "Stripe"),
])
def test_company_of(header, expected):
    assert company_of(header) == expected


@pytest.mark.parametrize("header, body, expected", [
    ("Acme | Engineer | REMOTE", "", "remote"),
    ("Acme | Engineer | NYC | Hybrid", "", "hybrid"),
    ("Acme | Engineer | ONSITE | SF", "", "onsite"),
    ("Acme | Engineer | SF | No remote", "", "onsite"),
    ("Acme | Engineer | Hybrid (no remote)", "", "hybrid"),
    ("Acme | Engineer | Remote or Onsite", "", "remote"),
    ("Acme | Engineer | Berlin", "We are fully remote friendly.", "remote"),
    ("Acme | Engineer | Berlin", "Great team.", "unspecified"),
])
def test_work_mode(header, body, expected):
    assert work_mode(header, body) == expected


def test_visa():
    assert offers_visa("VISA sponsorship available")
    assert offers_visa("Acme | SF | ONSITE | VISA")
    assert not offers_visa("Unfortunately we cannot sponsor visas")
    assert not offers_visa("No visa sponsorship")
    assert not offers_visa("Remote, US only")


def test_seniority():
    f = seniority_flags("Senior Backend Engineer, Staff ML Engineer, Interns welcome")
    assert f["lvl_senior"] and f["lvl_staff_principal"] and f["lvl_intern"]
    assert not f["lvl_junior"]
    assert seniority_flags("New grads welcome")["lvl_junior"]


def test_roles_from_titles_not_prose():
    header = "Acme | Machine Learning Engineer, Data Scientist | SF"
    body = header + "\nWe use machine learning across our whole product to delight customers every day."
    flags = role_flags(role_text(header, body))
    assert flags["role_ml_ai"] and flags["role_data_science"]
    assert not flags["role_frontend"]

    prose_only = "Acme | Backend Engineer | SF"
    flags = role_flags(role_text(prose_only, prose_only + "\n" + "We apply machine learning engineers' "
                                 "best ideas in a very long paragraph that is clearly prose and not a title."))
    assert not flags["role_ml_ai"]
    assert flags["role_backend"]


def test_ml_role_variants():
    for title in ["ML Engineer", "AI Engineer", "Applied Scientist", "LLM research engineer",
                  "MLE", "Research Scientist", "Senior AI/ML Engineer"]:
        assert role_flags(title)["role_ml_ai"], title
    assert not role_flags("HTML engineer")["role_ml_ai"]


@pytest.mark.parametrize("header, metro", [
    ("Acme | SF | Onsite", "SF Bay Area"),
    ("Acme | New York, NY or Remote", "New York"),
    ("Acme | London, UK | Hybrid", "London"),
    ("Acme | Palo Alto & NYC", "SF Bay Area"),
    ("Acme | Remote (US)", "Other/unspecified"),
])
def test_metro(header, metro):
    assert metro_of(header) == metro
