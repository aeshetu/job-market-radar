import pytest

from radar.skills import CATEGORY, SKILLS, extract_skills


def has(text, skill):
    return skill in extract_skills(text)


@pytest.mark.parametrize("text", [
    "Our stack: Go, Postgres, Kubernetes",
    "Backend in Go and Python",
    "We write Golang microservices",
    "Stack: Go/React",
    "Go (Golang) experience",
])
def test_go_positive(text):
    assert has(text, "Go")


@pytest.mark.parametrize("text", [
    "Go to our website to apply",
    "Go ahead and email us",
    "we're ready to go big",
    "go-to-market team",
    "Go! Apply now",
])
def test_go_negative(text):
    assert not has(text, "Go")


def test_c_language_vs_series_c():
    assert has("Experience with C and embedded Linux", "C")
    assert has("C/C++ developers", "C")
    assert not has("We just raised our Series C", "C")
    assert not has("We recently closed our series C and are hiring", "C")
    assert not has("C-level executives", "C")
    assert not has("Work with C# and .NET", "C")
    assert not has("Strong C++ skills", "C")


def test_cpp_and_csharp():
    assert has("Strong C++ skills", "C++")
    assert has("Work with C# and .NET", "C#")
    assert has("Work with C# and .NET", ".NET")


def test_r_language_vs_r_and_d():
    assert has("Python or R for analysis", "R")
    assert has("experience in R.", "R")
    assert not has("Our R&D team", "R")
    assert not has("R & D lab", "R")


def test_java_vs_javascript():
    assert has("Java, Spring Boot", "Java")
    assert not has("JavaScript and TypeScript", "Java")
    assert has("JavaScript and TypeScript", "JavaScript")


def test_react_vs_verb_and_native():
    assert has("React, Redux, Node.js", "React")
    assert not has("you'll react quickly to incidents", "React")
    assert not has("React Native mobile app", "React")
    assert has("React Native mobile app", "React Native")


def test_spark_requires_capital():
    assert has("Apache Spark pipelines", "Spark")
    assert has("pyspark jobs", "Spark")
    assert not has("ideas that spark joy", "Spark")


def test_llm_family():
    for text in ["Building LLM-powered apps", "experience with large language models",
                 "GPT-4 integrations", "generative AI startup", "GenAI platform"]:
        assert has(text, "LLMs / GenAI"), text


def test_machine_learning_case_sensitive_abbrev():
    assert has("ML infrastructure", "Machine learning")
    assert not has("html and xml parsing", "Machine learning")


def test_taxonomy_is_consistent():
    assert len(SKILLS) == len(set(SKILLS))
    assert set(CATEGORY) == set(SKILLS)
