#!/usr/bin/env python3
"""Notebook source for a local, paper-aligned HinPhish reproduction.

No official HinPhish training code was found. This notebook therefore builds a
local reimplementation around the feature schema described in the paper:

- extract website, domain, and resource link relationships
- classify relations as local / foreign / null / relative
- compute HIN-inspired phish-scores with iterative propagation
- train classical classifiers on the resulting feature table

The notebook assumes the Phishpedia benchmark archives will be present when it
runs. Because those archives are large, automatic extraction is off by default.
"""

from __future__ import annotations

# %%!
import subprocess
import sys

INSTALL_DEPS = False

if INSTALL_DEPS:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "beautifulsoup4",
            "numpy",
            "pandas",
            "scikit-learn",
            "tldextract",
        ]
    )

# %%!
import json
import math
import pickle
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import tldextract
from bs4 import BeautifulSoup
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# %%!
BASE_DIR = Path.cwd()
if not (BASE_DIR / "phish_sample_30k.zip").exists() and not (BASE_DIR / "DOWNLOAD_BLOCKED.md").exists():
    BASE_DIR = Path(__file__).resolve().parent

PHISH_ZIP = BASE_DIR / "phish_sample_30k.zip"
BENIGN_ZIP = BASE_DIR / "benign_sample_30k.zip"
PHISH_DIR = BASE_DIR / "phish_sample_30k"
BENIGN_DIR = BASE_DIR / "benign_sample_30k"
PREPARED_DIR = BASE_DIR / "prepared"
RUN_DIR = BASE_DIR / "runs" / "paper_reproduction"
MODEL_DIR = RUN_DIR / "models"

SEED = 3407
EXTRACT_ARCHIVES = False
MAX_SITES_PER_CLASS = 2000
MAX_ITER = 50
TOL = 1e-5

for path in (PREPARED_DIR, RUN_DIR, MODEL_DIR):
    path.mkdir(parents=True, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("PHISH_ZIP:", PHISH_ZIP)
print("BENIGN_ZIP:", BENIGN_ZIP)

extractor = tldextract.TLDExtract(suffix_list_urls=None)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def registrable_domain(value: str) -> str:
    extracted = extractor(value)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}".lower()
    if extracted.domain:
        return extracted.domain.lower()
    parsed = urlparse(value)
    return (parsed.netloc or parsed.path).lower()


def ensure_extracted(zip_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        return
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing archive: {zip_path}")
    if not EXTRACT_ARCHIVES:
        raise RuntimeError(
            f"{output_dir} does not exist. Either extract {zip_path.name} manually or set EXTRACT_ARCHIVES=True."
        )
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)


def iter_site_dirs(root: Path) -> list[Path]:
    site_dirs: list[Path] = []
    for info_path in root.rglob("info.txt"):
        site_dir = info_path.parent
        if (site_dir / "html.txt").exists():
            site_dirs.append(site_dir)
    site_dirs.sort()
    return site_dirs


def read_url_from_info(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("http://") or line.startswith("https://"):
            return line
    stripped = text.strip()
    if stripped:
        return stripped.splitlines()[0].strip()
    raise RuntimeError(f"Could not extract a URL from {path}")


def is_nullish(reference: str) -> bool:
    lowered = (reference or "").strip().lower()
    return lowered in {"", "#", "/", "javascript:", "javascript:void(0)", "javascript:void(0);", "about:blank"} or (
        lowered.startswith("javascript:") or lowered.startswith("mailto:") or lowered.startswith("tel:")
    )


def classify_reference(base_url: str, reference: str) -> tuple[str, str]:
    if is_nullish(reference):
        return "null", "<null>"

    reference = reference.strip()
    parsed_reference = urlparse(reference)

    if reference.startswith("//"):
        absolute = urlparse(urljoin(base_url, f"https:{reference}"))
    elif parsed_reference.scheme in {"http", "https"}:
        absolute = parsed_reference
    elif reference.startswith("#"):
        return "null", "<fragment>"
    else:
        return "relative", "<relative>"

    base_domain = registrable_domain(base_url)
    target_domain = registrable_domain(absolute.netloc)
    if target_domain == base_domain:
        return "local", target_domain
    return "foreign", target_domain


def relation_score(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    weighted = (
        counts["foreign"] * 1.0
        + counts["null"] * 1.0
        + counts["relative"] * 0.25
        + counts["local"] * -1.0
    )
    return max(-1.0, min(1.0, weighted / total))


def extract_hinphish_features(site_dir: Path, label_name: str) -> dict[str, object]:
    url = read_url_from_info(site_dir / "info.txt")
    html = (site_dir / "html.txt").read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    domain_objects: dict[str, Counter] = defaultdict(Counter)
    resource_objects: dict[str, Counter] = defaultdict(Counter)

    domain_tags = [
        ("a", "href"),
        ("form", "action"),
    ]
    resource_tags = [
        ("script", "src"),
        ("img", "src"),
        ("link", "href"),
        ("iframe", "src"),
        ("source", "src"),
        ("embed", "src"),
    ]

    for tag_name, attr_name in domain_tags:
        for tag in soup.find_all(tag_name):
            reference = tag.get(attr_name)
            relation, object_key = classify_reference(url, reference or "")
            domain_objects[object_key][relation] += 1

    for tag_name, attr_name in resource_tags:
        for tag in soup.find_all(tag_name):
            reference = tag.get(attr_name)
            relation, object_key = classify_reference(url, reference or "")
            resource_objects[object_key][relation] += 1

    all_objects: dict[tuple[str, str], Counter] = {}
    for key, counts in domain_objects.items():
        all_objects[("domain", key)] = counts
    for key, counts in resource_objects.items():
        all_objects[("resource", key)] = counts

    initial_scores = {node: relation_score(counts) for node, counts in all_objects.items()}
    current_scores = initial_scores.copy()
    iterations = 0

    for iterations in range(1, MAX_ITER + 1):
        page_score = float(np.mean(list(current_scores.values()))) if current_scores else 0.0
        new_scores = {}
        max_delta = 0.0
        for node, base_score in initial_scores.items():
            propagated = 0.6 * base_score + 0.4 * page_score
            propagated = max(-1.0, min(1.0, propagated))
            new_scores[node] = propagated
            max_delta = max(max_delta, abs(propagated - current_scores[node]))
        current_scores = new_scores
        if max_delta < TOL:
            break

    domain_scores = [score for (node_type, _), score in current_scores.items() if node_type == "domain"]
    resource_scores = [score for (node_type, _), score in current_scores.items() if node_type == "resource"]
    all_scores = list(current_scores.values())

    def count_relations(objects: dict[str, Counter], relation: str) -> int:
        return int(sum(counter[relation] for counter in objects.values()))

    row = {
        "site_dir": str(site_dir),
        "url": url,
        "label_name": label_name,
        "label": 1 if label_name == "phishing" else 0,
        "score": float(np.mean(all_scores)) if all_scores else 0.0,
        "iters": int(iterations),
        "dom_local_link": count_relations(domain_objects, "local"),
        "dom_foreign_link": count_relations(domain_objects, "foreign"),
        "dom_null_link": count_relations(domain_objects, "null"),
        "dom_relative_link": count_relations(domain_objects, "relative"),
        "res_local_link": count_relations(resource_objects, "local"),
        "res_foreign_link": count_relations(resource_objects, "foreign"),
        "res_null_link": count_relations(resource_objects, "null"),
        "res_relative_link": count_relations(resource_objects, "relative"),
        "dom_mean": float(np.mean(domain_scores)) if domain_scores else 0.0,
        "dom_var": float(np.var(domain_scores)) if domain_scores else 0.0,
        "mean": float(np.mean(all_scores)) if all_scores else 0.0,
        "var": float(np.var(all_scores)) if all_scores else 0.0,
        "domain_object_count": int(len(domain_objects)),
        "resource_object_count": int(len(resource_objects)),
        "resource_mean": float(np.mean(resource_scores)) if resource_scores else 0.0,
        "resource_var": float(np.var(resource_scores)) if resource_scores else 0.0,
    }
    return row


# %%!
ensure_extracted(PHISH_ZIP, PHISH_DIR)
ensure_extracted(BENIGN_ZIP, BENIGN_DIR)

phish_sites = iter_site_dirs(PHISH_DIR)
benign_sites = iter_site_dirs(BENIGN_DIR)

rng = random.Random(SEED)
rng.shuffle(phish_sites)
rng.shuffle(benign_sites)

if MAX_SITES_PER_CLASS:
    phish_sites = phish_sites[:MAX_SITES_PER_CLASS]
    benign_sites = benign_sites[:MAX_SITES_PER_CLASS]

site_summary = {
    "phishing_site_dirs": len(phish_sites),
    "benign_site_dirs": len(benign_sites),
    "max_sites_per_class": MAX_SITES_PER_CLASS,
}
write_json(RUN_DIR / "site_summary.json", site_summary)
print(json.dumps(site_summary, indent=2))

# %%!
rows = []
for site_dir in phish_sites:
    rows.append(extract_hinphish_features(site_dir, "phishing"))
for site_dir in benign_sites:
    rows.append(extract_hinphish_features(site_dir, "benign"))

feature_df = pd.DataFrame(rows)
feature_df.to_csv(PREPARED_DIR / "hinphish_features.csv", index=False)

summary = {
    "rows": int(len(feature_df)),
    "label_distribution": feature_df["label_name"].value_counts().to_dict(),
    "columns": feature_df.columns.tolist(),
}
write_json(RUN_DIR / "data_summary.json", summary)
print(json.dumps(summary, indent=2))

# %%!
feature_columns = [
    "score",
    "iters",
    "dom_local_link",
    "dom_foreign_link",
    "dom_null_link",
    "dom_relative_link",
    "res_local_link",
    "res_foreign_link",
    "res_null_link",
    "res_relative_link",
    "dom_mean",
    "dom_var",
    "mean",
    "var",
    "domain_object_count",
    "resource_object_count",
    "resource_mean",
    "resource_var",
]

X = feature_df[feature_columns].fillna(0.0)
y = feature_df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)

models = {
    "logistic_regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=4000)),
        ]
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=400,
        random_state=SEED,
        class_weight="balanced",
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
}

results: list[dict[str, object]] = []
best_name = None
best_model = None
best_f1 = -1.0

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }
    results.append(metrics)
    write_json(RUN_DIR / f"{model_name}_metrics.json", metrics)
    if metrics["f1"] > best_f1:
        best_f1 = metrics["f1"]
        best_name = model_name
        best_model = model

leaderboard = pd.DataFrame(results).drop(columns=["classification_report"]).sort_values(
    by=["f1", "accuracy"],
    ascending=False,
)
leaderboard.to_csv(RUN_DIR / "leaderboard.csv", index=False)

if best_model is None or best_name is None:
    raise RuntimeError("No HinPhish model was trained.")

with (MODEL_DIR / f"{best_name}.pkl").open("wb") as handle:
    pickle.dump(best_model, handle)

best_summary = next(item for item in results if item["model"] == best_name)
write_json(
    RUN_DIR / "best_model_summary.json",
    {
        "best_model": best_name,
        "best_f1": best_f1,
        "feature_columns": feature_columns,
        "metrics": best_summary,
    },
)

print(leaderboard)
print("Best model:", best_name)
