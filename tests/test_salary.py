import pytest

from radar.salary import extract_salary, find_salaries


@pytest.mark.parametrize("text, low, high, cur", [
    ("Senior Engineer | $120k - $160k + equity", 120_000, 160_000, "USD"),
    ("Salary: $120,000 – $160,000", 120_000, 160_000, "USD"),
    ("$200-300K base", 200_000, 300_000, "USD"),
    ("$150K-180K", 150_000, 180_000, "USD"),
    ("comp 120-150k USD", 120_000, 150_000, "USD"),
    ("€80.000 - 95.000 per year", 80_000, 95_000, "EUR"),
    ("£60k to £80k", 60_000, 80_000, "GBP"),
    ("CA$110k - CA$140k", 110_000, 140_000, "CAD"),
    ("Pay: $185k+", 185_000, 185_000, "USD"),
    ("base salary $175,000", 175_000, 175_000, "USD"),
    ("$150.5k-$170k", 150_500, 170_000, "USD"),
    ("Compensation: $45,000 - $55,000 CAD plus benefits", 45_000, 55_000, "CAD"),
    ("Salary from $120k to $150k", 120_000, 150_000, "USD"),
    ("Engineers earn $150k-$200k plus equity", 150_000, 200_000, "USD"),
])
def test_parses_annual_salaries(text, low, high, cur):
    s = extract_salary(text)
    assert (s["salary_low"], s["salary_high"], s["salary_currency"]) == (low, high, cur)


@pytest.mark.parametrize("text", [
    "We raised $10-20M from great investors",
    "We just raised $25M Series B",
    "Contract role, $80-100/hr",
    "$60 - $90 per hour",
    "€5k-7k per month",
    "$5,000-$7,000/month stipend",
    "Founded 2015-2020, 50-100 employees",
    "$50,000 signing bonus",
    "We have $2M ARR",
    "0.1% - 0.5% equity",
    "I grew my last SaaS from $0 to $33,000 MRR",
    "accessible to the 90M Americans making $40,000 or less",
    "YC invested ~$17k in 2010",
])
def test_rejects_non_salaries(text):
    assert find_salaries(text) == []


def test_multiple_ranges_span_and_median():
    s = extract_salary("Backend $140k-$180k, Staff $200k-$260k")
    assert s["salary_low"] == 140_000
    assert s["salary_high"] == 260_000
    assert s["salary_mid"] == (160_000 + 230_000) / 2
    assert s["salary_mentions"] == 2


def test_dominant_currency_wins():
    s = extract_salary("$150k-$180k (US) or £90k-£110k (UK), US$140k-160k")
    assert s["salary_currency"] == "USD"
    assert s["salary_low"] == 140_000


def test_no_salary():
    assert extract_salary("Great team, competitive pay")["salary_mid"] is None
