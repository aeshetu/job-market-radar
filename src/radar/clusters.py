"""Unsupervised map of *what kinds of companies* post on HN, and how that shifts.

1. K-means on MiniLM embeddings (k checked with silhouette on a sample).
2. Zero-shot cluster naming: each centroid is matched to the nearest plain-English
   industry description embedded with the same model. Centroids that aren't close to
   any description are labelled "General: <keywords>" rather than forced into a
   category. c-TF-IDF keywords are stored alongside so every name can be checked.
3. Monthly cluster shares, a 2-D UMAP map of a sample, and a nearest-neighbour index
   of the last 12 months of posts for the dashboard's resume matcher.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

from . import embed
from .config import INTERIM, PROCESSED, RANDOM_STATE
from .parse import strip_urls

log = logging.getLogger(__name__)

K = 24
K_GRID = (12, 16, 20, 24, 28, 32)
NAME_THRESHOLD = 0.5  # cosine; below this a centroid isn't close to any industry description
MAP_SAMPLE = 12_000
MATCH_MONTHS = 12

DOMAINS = {
    "Fintech & payments": "a fintech company building payments, banking, lending or financial software",
    "Trading & quant finance": "a hedge fund or trading firm doing quantitative and algorithmic trading",
    "Crypto & Web3": "a cryptocurrency, blockchain, web3 or DeFi company",
    "Healthcare & digital health": "a healthcare company improving patient care, clinics, doctors and health insurance",
    "Biotech & life sciences": "a biotech company doing drug discovery, genomics and life sciences research",
    "AI / LLM products": "an AI startup building products with large language models and generative AI agents",
    "ML research & infrastructure": "a company doing machine learning research, training models and GPU infrastructure",
    "Developer tools": "a developer tools company building software for programmers, APIs and open source",
    "Cloud & infrastructure": "a cloud infrastructure company building databases, distributed systems and DevOps platforms",
    "Data & analytics": "a data analytics platform for business intelligence and data pipelines",
    "Cybersecurity": "a cybersecurity company protecting networks, identity and preventing attacks",
    "E-commerce & retail": "an e-commerce and retail company selling products online with marketplaces",
    "Marketing & adtech": "an advertising technology and marketing automation company",
    "Media & content": "a media, news, publishing, video and streaming content company",
    "Social & consumer apps": "a consumer social network and community startup",
    "Mobile apps": "a company building iOS and Android mobile apps",
    "Gaming": "a video game studio building games",
    "Education": "an education technology company for students, schools and online learning",
    "HR & recruiting": "an HR, recruiting, hiring and workforce management platform",
    "Enterprise SaaS": "a B2B SaaS company selling software to businesses and enterprise customers",
    "Real estate & proptech": "a real estate, property and housing technology company",
    "Logistics & supply chain": "a logistics, shipping, freight and supply chain company",
    "Mobility & transportation": "a transportation company working on cars, autonomous vehicles, ride sharing and travel",
    "Climate & energy": "a climate tech and clean energy company working on solar, batteries and carbon",
    "Robotics & hardware": "a robotics and hardware company building physical devices, sensors and manufacturing",
    "Aerospace & defense": "an aerospace, space, satellite and defense technology company",
    "Government & civic tech": "a government, public sector and civic technology organization",
    "Legal & compliance": "a legal technology and regulatory compliance software company",
    "Insurance": "an insurance technology company",
    "Food & hospitality": "a food, restaurant, delivery and hospitality company",
    "Agencies & consulting": "a software consultancy or agency building apps for clients",
    "Nonprofit & research": "a nonprofit, university lab or research institute",
    "Travel": "a travel booking, hotels and tourism company",
}


def center(x: np.ndarray) -> np.ndarray:
    """Subtract the mean vector and re-normalise.

    Every job post shares a strong "this is a hiring ad" direction (benefits, team,
    apply, ability to communicate). Left in, it dominates cosine similarity and pulls
    generic clusters toward descriptions like "HR & recruiting". Centering (as in
    'all-but-the-top' post-processing) keeps what distinguishes posts from each other.
    Industry descriptions are centered on their own mean for the same reason.
    """
    x = x - x.mean(axis=0, keepdims=True)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def choose_k(emb: np.ndarray, rng: np.random.Generator) -> dict[int, float]:
    idx = rng.choice(len(emb), size=min(15_000, len(emb)), replace=False)
    out = {}
    for k in K_GRID:
        km = KMeans(n_clusters=k, n_init=3, random_state=RANDOM_STATE).fit(emb[idx])
        out[k] = float(silhouette_score(emb[idx], km.labels_, sample_size=8000, random_state=RANDOM_STATE))
    return out


def ctfidf_keywords(texts: pd.Series, labels: np.ndarray, companies: pd.Series,
                    top: int = 10) -> dict[int, list[str]]:
    """Class-based TF-IDF (as in BERTopic): which words are distinctive for each cluster.

    Each company counts once per cluster, using binary term presence. Otherwise a
    company that reposts for 40 months turns its own name into the top keyword.
    """
    d = pd.DataFrame({"text": texts.to_numpy(), "label": labels, "company": companies.to_numpy()})
    d = d.drop_duplicates(["label", "company"])
    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=25, max_df=0.3, binary=True,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z+#.-]{2,}\b")
    counts = vec.fit_transform(d["text"].map(strip_urls))
    vocab = np.array(vec.get_feature_names_out())
    lab = d["label"].to_numpy()
    ks = np.unique(labels)
    tf = np.vstack([np.asarray(counts[lab == k].sum(axis=0)).ravel() for k in ks]).astype(float)
    tf /= tf.sum(axis=1, keepdims=True)
    idf = np.log(1 + tf.mean(axis=1).mean() / (tf.mean(axis=0) + 1e-12))
    score = tf * idf
    return {int(k): vocab[np.argsort(-score[i])[:top]].tolist() for i, k in enumerate(ks)}


def name_clusters(centroids: np.ndarray, model, keywords: dict[int, list[str]]) -> tuple[list[str], np.ndarray]:
    """Nearest industry description per centroid, if it's close enough.

    Clusters below NAME_THRESHOLD are *not* industries (they group generic posts),
    so they're labelled "General: <keywords>" instead of being forced into a
    category. Repeated names get their top keyword appended.
    """
    label_emb = center(embed.encode(model, list(DOMAINS.values())))
    c = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    sim = c @ label_emb.T
    best = sim.argmax(axis=1)
    top_sim = sim[np.arange(len(c)), best]
    names = [list(DOMAINS)[b] if s >= NAME_THRESHOLD else "General: " + ", ".join(keywords[i][:3])
             for i, (b, s) in enumerate(zip(best, top_sim))]
    counts = pd.Series(names).value_counts()
    names = [f"{n} ({keywords[i][0]})" if counts[n] > 1 else n for i, n in enumerate(names)]
    return names, top_sim


def run() -> None:
    posts = pd.read_parquet(INTERIM / "posts.parquet")
    ids, emb = embed.load()
    pos = pd.Series(np.arange(len(ids)), index=ids)
    posts = posts[posts["post_id"].isin(ids)].reset_index(drop=True)
    emb = emb[pos.reindex(posts["post_id"]).to_numpy()]
    rng = np.random.default_rng(RANDOM_STATE)
    raw_emb = emb
    emb = center(emb)

    sil = choose_k(emb, rng)
    log.info("Silhouette by k: %s", sil)
    km = KMeans(n_clusters=K, n_init=5, random_state=RANDOM_STATE).fit(emb)
    labels = km.labels_.astype(np.int64)
    np.save(INTERIM / "cluster_labels.npy", labels)
    model = embed.load_model()
    keywords = ctfidf_keywords(posts["text"].map(embed.doc_text), labels, posts["company_key"])
    log.info("Keywords: %s", {k: v[:4] for k, v in keywords.items()})
    names, name_sim = name_clusters(km.cluster_centers_, model, keywords)

    info = pd.DataFrame({
        "cluster": range(K), "name": names, "name_similarity": name_sim,
        "keywords": [", ".join(keywords[k][:8]) for k in range(K)],
        "size": np.bincount(labels, minlength=K),
    })
    # Example headers closest to each centroid make a name easy to verify by eye.
    dist = km.transform(emb)
    info["examples"] = [" || ".join(posts["header"].iloc[np.argsort(dist[:, k])[:4]].str[:90]) for k in range(K)]
    info.to_parquet(PROCESSED / "clusters.parquet", index=False)

    posts["cluster"] = labels
    cm = posts.groupby(["month", "cluster"]).size().rename("n").reset_index()
    cm = cm.merge(posts.groupby("month").size().rename("posts").reset_index(), on="month")
    cm["share"] = cm["n"] / cm["posts"]
    cm["name"] = cm["cluster"].map(dict(enumerate(names)))
    cm.to_parquet(PROCESSED / "cluster_monthly.parquet", index=False)

    # 2-D map of a stratified-by-time sample.
    import umap
    per_year = MAP_SAMPLE // posts["year"].nunique()
    sample = posts.sample(frac=1.0, random_state=RANDOM_STATE).groupby("year").head(per_year)
    xy = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                   random_state=RANDOM_STATE).fit_transform(emb[sample.index])
    pd.DataFrame({
        "post_id": sample["post_id"].to_numpy(), "x": xy[:, 0], "y": xy[:, 1],
        "year": sample["year"].to_numpy(), "cluster": sample["cluster"].to_numpy(),
        "name": sample["cluster"].map(dict(enumerate(names))).to_numpy(),
        "header": sample["header"].str[:110].to_numpy(),
    }).to_parquet(PROCESSED / "cluster_map.parquet", index=False)

    # Resume-matcher index: recent posts only, float16 to keep the repo small.
    recent = posts[posts["month"] > posts["month"].max() - pd.DateOffset(months=MATCH_MONTHS)]
    recent[["post_id", "month", "company", "header", "work_mode", "metro", "skills",
            "salary_low", "salary_high", "salary_currency", "cluster"]].to_parquet(
        PROCESSED / "match_posts.parquet", index=False)
    np.save(PROCESSED / "match_embeddings.npy", raw_emb[recent.index].astype(np.float16))

    (PROCESSED / "cluster_metrics.json").write_text(json.dumps({
        "k": K, "silhouette_by_k": sil, "n_posts": int(len(posts)),
        "mean_name_similarity": float(np.mean(name_sim)),
    }, indent=2))
    log.info("Clusters:\n%s", info[["name", "size", "keywords"]].to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
