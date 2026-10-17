"""Skill taxonomy and extraction.

Each skill is a regex instead of a plain keyword, because the naive versions fail badly:
"Go" (go to our site), "C" (Series C), "R" (R&D), "React" (react quickly),
"Spark" (spark joy), "Java" vs "JavaScript". Case-sensitive patterns are used
where capitalisation carries the meaning. tests/test_skills.py pins these cases.
"""
from __future__ import annotations

import re

# (name, category, pattern, case_sensitive)
_GO_STOP = (r"to|ahead|back|beyond|live|further|forward|through|from|get|check|out|after|"
            r"the|a|an|into|over|anywhere|home|public|global|remote|fast|faster|big|deep|wild|"
            r"above|far|hand|together|places|all|about|green|figure|see|and see")

SKILL_DEFS: list[tuple[str, str, str, bool]] = [
    # --- Languages
    ("Python", "Language", r"\bpython\b", False),
    ("Java", "Language", r"\bjava\b(?!\s?script)", False),
    ("JavaScript", "Language", r"\bjavascript\b|\bES6\b|\bECMAScript\b", False),
    ("TypeScript", "Language", r"\btypescript\b", False),
    ("Go", "Language", rf"\bgolang\b|\bGolang\b|(?<![\w.#/@-])Go(?![\w#+'’-])(?!!)(?!\s+(?:{_GO_STOP})\b)", True),
    ("Rust", "Language", r"\bRust\b|\brust(?=,|/| and| or|\))", True),
    ("C++", "Language", r"(?<![\w+])C\+\+(?!\w)|\bcpp\b", False),
    ("C#", "Language", r"(?<![\w#])C#(?![\w#])", False),
    ("C", "Language", r"(?<![\w+#&./'’-])(?<![Ss]eries )(?<![Rr]ound )(?<![Cc]lass )(?<![Tt]ype )"
                      r"(?<![Pp]lan )(?<![Vv]itamin )(?<![Ss]eries-)"
                      r"C(?![\w+#&'’./-])(?!\s?(?:level|suite|corp)\b)|\bC/C\+\+", True),
    ("Ruby", "Language", r"\bruby\b", False),
    ("PHP", "Language", r"\bphp\b", False),
    ("Scala", "Language", r"\bscala\b", False),
    ("Kotlin", "Language", r"\bkotlin\b", False),
    ("Swift", "Language", r"\bSwift(?:UI)?\b", True),
    ("Objective-C", "Language", r"\bobjective[- ]?c\b|\bobj-?c\b", False),
    ("Elixir", "Language", r"\belixir\b", False),
    ("Erlang", "Language", r"\berlang\b", False),
    ("Haskell", "Language", r"\bhaskell\b", False),
    ("Clojure", "Language", r"\bclojure(?:script)?\b", False),
    ("OCaml", "Language", r"\bocaml\b", False),
    ("Perl", "Language", r"\bperl\b", False),
    ("R", "Language", r"(?<![\w&/.'’-])R(?![\w&/'’+-])(?!\.\w)(?!\s?&\s?D)", True),
    ("Lua", "Language", r"\blua\b", False),
    ("Zig", "Language", r"\bZig\b", True),
    ("Solidity", "Language", r"\bsolidity\b", False),
    ("SQL", "Language", r"\bSQL\b", False),
    # --- Web / app frameworks
    ("React", "Framework", r"\bReact(?:\.?js|JS)?\b(?![\s-]?Native)|\breact\.?js\b", True),
    ("React Native", "Mobile", r"\breact[\s-]?native\b", False),
    ("Angular", "Framework", r"\bangular(?:\.?js|JS)?\b", False),
    ("Vue", "Framework", r"\bvue(?:\.?js)?\b", False),
    ("Svelte", "Framework", r"\bsvelte(?:kit)?\b", False),
    ("Next.js", "Framework", r"\bnext\.?js\b", False),
    ("Node.js", "Framework", r"\bNode(?:\.?[jJ][sS])?\b|\bnode\.?js\b", True),
    ("jQuery", "Framework", r"\bjquery\b", False),
    ("Backbone.js", "Framework", r"\bbackbone\.?js\b|\bBackbone\b(?! of)", True),
    ("Ember", "Framework", r"\bember(?:\.?js)?\b", False),
    ("Django", "Framework", r"\bdjango\b", False),
    ("Flask", "Framework", r"\bflask\b", False),
    ("FastAPI", "Framework", r"\bfast\s?api\b", False),
    ("Rails", "Framework", r"\bRails\b|\bRoR\b|\bruby on rails\b", True),
    ("Spring", "Framework", r"\bspring[\s-]?boot\b|\bspring (?:mvc|framework|cloud)\b", False),
    (".NET", "Framework", r"(?<![\w.])\.NET\b|\bdotnet\b|\bASP\.NET\b", False),
    ("Laravel", "Framework", r"\blaravel\b", False),
    ("GraphQL", "Framework", r"\bgraphql\b", False),
    ("gRPC", "Framework", r"\bgrpc\b", False),
    # --- Data & ML
    ("Machine learning", "Data & ML", r"\bmachine[- ]learning\b|\bML\b", True),
    ("Deep learning", "Data & ML", r"\bdeep[- ]learning\b|\bneural net(?:work)?s?\b", False),
    ("AI (any mention)", "Data & ML", r"\bAI\b|\bartificial intelligence\b", True),
    ("LLMs / GenAI", "Data & ML", r"\bLLMs?\b|\blarge language models?\b|\bGPT-?\d\w*\b|\bChatGPT\b"
                                   r"|\bgen(?:erative)?[\s-]?AI\b|\bGenAI\b|\bfoundation models?\b", False),
    ("AI agents", "Data & ML", r"\bAI agents?\b|\bagentic\b|\bLLM agents?\b|\bcoding agents?\b", False),
    ("RAG", "Data & ML", r"\bRAG\b|\bretrieval[- ]augmented\b", True),
    ("Vector DBs", "Data & ML", r"\bvector (?:database|db|search|store)s?\b|\bpinecone\b|\bpgvector\b"
                                 r"|\bweaviate\b|\bqdrant\b|\bmilvus\b", False),
    ("NLP", "Data & ML", r"\bNLP\b|\bnatural language processing\b", False),
    ("Computer vision", "Data & ML", r"\bcomputer vision\b", False),
    ("MLOps", "Data & ML", r"\bmlops\b", False),
    ("PyTorch", "Data & ML", r"\bpy\s?torch\b", False),
    ("TensorFlow", "Data & ML", r"\btensor\s?flow\b", False),
    ("Keras", "Data & ML", r"\bkeras\b", False),
    ("scikit-learn", "Data & ML", r"\bscikit[- ]?learn\b|\bsklearn\b", False),
    ("Hugging Face", "Data & ML", r"\bhugging\s?face\b", False),
    ("OpenAI", "Data & ML", r"\bopenai\b", False),
    ("LangChain", "Data & ML", r"\blangchain\b|\blanggraph\b", False),
    ("pandas", "Data & ML", r"\bpandas\b", False),
    ("NumPy", "Data & ML", r"\bnumpy\b", False),
    ("Spark", "Data & ML", r"\b(?:Apache )?Spark\b|\bPySpark\b|\bpyspark\b", True),
    ("Hadoop", "Data & ML", r"\bhadoop\b|\bmapreduce\b", False),
    ("Kafka", "Data & ML", r"\bkafka\b", False),
    ("Airflow", "Data & ML", r"\bairflow\b", False),
    ("dbt", "Data & ML", r"\bdbt\b", False),
    ("Snowflake", "Data & ML", r"\bSnowflake\b", True),
    ("Databricks", "Data & ML", r"\bdatabricks\b", False),
    ("BigQuery", "Data & ML", r"\bbig\s?query\b", False),
    ("Redshift", "Data & ML", r"\bredshift\b", False),
    # --- Databases
    ("PostgreSQL", "Database", r"\bpostgres(?:ql)?\b|\bpsql\b", False),
    ("MySQL", "Database", r"\bmysql\b|\bmariadb\b", False),
    ("MongoDB", "Database", r"\bmongo(?:db)?\b", False),
    ("Redis", "Database", r"\bredis\b", False),
    ("Elasticsearch", "Database", r"\belastic\s?search\b|\bopensearch\b", False),
    ("Cassandra", "Database", r"\bcassandra\b", False),
    ("DynamoDB", "Database", r"\bdynamo\s?db\b", False),
    ("SQLite", "Database", r"\bsqlite\b", False),
    ("ClickHouse", "Database", r"\bclickhouse\b", False),
    # --- Cloud & infra
    ("AWS", "Cloud & Infra", r"\bAWS\b|\bamazon web services\b|\bEC2\b", True),
    ("GCP", "Cloud & Infra", r"\bGCP\b|\bgoogle cloud\b|\bGCE\b|\bapp engine\b", False),
    ("Azure", "Cloud & Infra", r"\bazure\b", False),
    ("Docker", "Cloud & Infra", r"\bdocker\b", False),
    ("Kubernetes", "Cloud & Infra", r"\bkubernetes\b|\bk8s\b", False),
    ("Terraform", "Cloud & Infra", r"\bterraform\b", False),
    ("Heroku", "Cloud & Infra", r"\bheroku\b", False),
    ("Linux", "Cloud & Infra", r"\blinux\b", False),
    ("Serverless", "Cloud & Infra", r"\bserverless\b|\b(?:AWS )?Lambda\b", True),
    # --- Mobile
    ("iOS", "Mobile", r"\biOS\b", True),
    ("Android", "Mobile", r"\bandroid\b", False),
    ("Flutter", "Mobile", r"\bflutter\b", False),
    # --- Domains
    ("Crypto / Web3", "Domain", r"\bblockchain\b|\bcrypto(?:currency|currencies)?\b|\bweb3\b|\bethereum\b"
                                 r"|\bbitcoin\b|\bsmart contracts?\b|\bdefi\b", False),
]

SKILLS = [name for name, *_ in SKILL_DEFS]
CATEGORY = {name: cat for name, cat, *_ in SKILL_DEFS}
_COMPILED = [
    (name, re.compile(pat, 0 if cs else re.IGNORECASE)) for name, _, pat, cs in SKILL_DEFS
]


def extract_skills(text: str) -> list[str]:
    """Skills mentioned in `text` (call parse.strip_urls first), in taxonomy order."""
    return [name for name, rx in _COMPILED if rx.search(text)]
