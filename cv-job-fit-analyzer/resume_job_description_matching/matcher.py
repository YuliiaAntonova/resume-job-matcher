# pip install spacy pdfplumber scikit-learn docx2txt
# python -m spacy download en_core_web_sm

import csv
import re
from typing import List, Dict, Tuple, Set

import pdfplumber
import docx2txt
import spacy
from spacy.matcher import PhraseMatcher

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# 1) spaCy model
# ----------------------------

nlp = spacy.load("en_core_web_sm")

# ----------------------------
# 2) Target roles: DE + DS + MLOps + DevOps
# ----------------------------

# We keep these roles; we do NOT filter them out.
ROLE_SIGNALS = [
    # Data roles
    "data engineer", "data engineering", "analytics engineer",
    "data scientist",
    "machine learning engineer", "ml engineer",
    "mlops", "ml ops", "ai ops", "aiops",
    "data analyst",  # optional, but user wants to keep DS/ML Ops/DevOps, analyst often close
    # DevOps
    "devops", "platform engineer", "site reliability", "sre",
]

# Hard excludes (we do NOT want C++ / embedded style jobs)
HARD_EXCLUDE_TITLE_PATTERNS = [
    r"\bc\+\+\b", r"\bcpp\b", r"\bqt\b",
    r"\bembedded\b", r"\bfirmware\b", r"\bdriver\b", r"\bkernel\b",
    r"\bdevice driver\b", r"\bmicrocontroller\b", r"\brtos\b",
]

# ----------------------------
# 3) Skills dictionary (targeted for DE/DS/MLOps/DevOps)
# ----------------------------

TECH_PHRASES = [
    # Languages
    "Python", "SQL", "Scala", "Java", "Go", "Bash",

    # Data Engineering
    "ETL", "ELT", "data pipeline", "data pipelines",
    "data warehouse", "DWH", "data lake", "lakehouse",
    "data modeling", "dimensional modeling", "Kimball", "star schema",
    "CDC", "SCD", "slowly changing dimensions",
    "data quality", "data validation", "data lineage",

    # Spark ecosystem
    "Spark", "PySpark", "Spark SQL",

    # Orchestration / Transform
    "Airflow", "Apache Airflow", "dbt",

    # Streaming
    "Kafka", "Kinesis", "Flink", "Apache Flink",

    # Lakehouse / formats
    "Iceberg", "Apache Iceberg", "Delta Lake", "Hudi",
    "Parquet", "ORC", "Avro",

    # Platforms / Warehouses
    "Databricks",
    "Snowflake", "BigQuery", "Google BigQuery",
    "Redshift", "Athena", "Trino", "Presto",

    # AWS / cloud
    "AWS", "Amazon Web Services",
    "AWS Glue", "Glue",
    "AWS Step Functions", "Step Functions", "StepFunctions",
    "Lambda", "S3", "EC2", "EMR",
    "DynamoDB", "CloudWatch",
    "SQS", "SNS",

    # MLOps / DS tooling
    "MLflow", "Kubeflow",
    "SageMaker", "Amazon SageMaker",
    "TensorFlow", "PyTorch",
    "scikit-learn", "sklearn",

    # DevOps / infra
    "Docker", "Kubernetes", "K8s", "Terraform",
    "CI/CD", "CI CD", "CICD",
    "Git", "GitHub Actions", "Jenkins",
    "Prometheus", "Grafana",

    # Databases / search
    "PostgreSQL", "Postgres", "MySQL", "Oracle",
    "OpenSearch", "Elasticsearch",
]

CANON_MAP = {
    "pyspark": "spark",
    "spark sql": "spark",
    "apache airflow": "airflow",
    "aws glue": "glue",
    "glue": "glue",
    "aws step functions": "step functions",
    "stepfunctions": "step functions",
    "amazon web services": "aws",
    "postgres": "postgresql",
    "apache iceberg": "iceberg",
    "delta lake": "delta",
    "apache flink": "flink",
    "k8s": "kubernetes",
    "cicd": "ci/cd",
    "ci cd": "ci/cd",
    "google bigquery": "bigquery",
    "amazon sagemaker": "sagemaker",
    "sklearn": "scikit-learn",
}

def canonicalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("–", "-").replace("—", "-")
    return CANON_MAP.get(s, s)

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
matcher.add("TECH", [nlp.make_doc(p) for p in TECH_PHRASES])

ALLOW = {
    # languages
    "python", "sql", "scala", "java", "go", "bash",

    # DE
    "etl", "elt", "cdc", "scd",
    "pipeline", "data pipeline", "data pipelines",
    "data warehouse", "dwh", "data lake", "lakehouse",
    "data modeling", "dimensional modeling", "kimball", "star schema",
    "data quality", "data validation", "data lineage",

    # spark
    "spark",

    # orchestration/transform
    "airflow", "dbt",

    # streaming
    "kafka", "kinesis", "flink",

    # lakehouse / formats
    "iceberg", "delta", "hudi", "parquet", "orc", "avro",

    # platforms
    "databricks", "snowflake", "bigquery", "trino", "presto",

    # aws
    "aws", "glue", "athena", "lambda", "s3", "ec2", "emr", "redshift", "dynamodb", "cloudwatch",
    "sqs", "sns",

    # mlops/ds
    "mlflow", "kubeflow", "sagemaker", "tensorflow", "pytorch", "scikit-learn",

    # devops/infra
    "docker", "kubernetes", "terraform", "ci/cd", "git", "jenkins", "github actions",
    "prometheus", "grafana",

    # db/search
    "postgresql", "mysql", "oracle", "opensearch", "elasticsearch",
}

def extract_skills(text: str) -> Set[str]:
    t = text or ""
    doc = nlp(t)
    skills: Set[str] = set()

    # A) phrase matcher
    for _, start, end in matcher(doc):
        skills.add(canonicalize(doc[start:end].text))

    # B) regex tokens (plus hyphenated)
    regex_hits = []
    regex_hits += re.findall(r"\b[A-Za-z]{2,}(?:/[A-Za-z]{2,})?\b|\b[A-Z]{2,}\d?\b", t)
    regex_hits += re.findall(r"\b[A-Za-z]{2,}(?:-[A-Za-z]{2,})+\b", t)

    for w in regex_hits:
        w2 = canonicalize(w)
        if w2 in ALLOW:
            skills.add(w2)

    return skills


# ----------------------------
# 4) Gating: keep target roles, exclude C++/embedded
# ----------------------------

# "Positive" signals that mean this is likely in our target area.
TARGET_AREA_SIGNALS = [
    # roles
    *ROLE_SIGNALS,

    # tech signals (DE/DS/MLOps/DevOps)
    "spark", "pyspark", "airflow", "dbt",
    "etl", "elt", "pipeline", "data pipeline",
    "kafka", "kinesis",
    "aws glue", "glue", "emr", "athena", "redshift",
    "snowflake", "bigquery", "databricks",
    "mlops", "ml ops", "kubeflow", "mlflow", "sagemaker",
    "kubernetes", "docker", "terraform", "ci/cd",
    "python", "sql", "scala",
]

# Anti signals in text (soft)
ANTI_SIGNALS = [
    "c++", "cpp", "qt",
    "embedded", "firmware", "microcontroller", "rtos",
    "kernel", "device driver", "drivers",
]

def count_signals(text: str, signals: List[str]) -> int:
    t = (text or "").lower()
    return sum(1 for s in signals if s in t)

def hard_exclude_by_title(title: str) -> bool:
    t = (title or "").lower()
    return any(re.search(p, t) for p in HARD_EXCLUDE_TITLE_PATTERNS)

def is_target_job(title: str, desc: str, min_target_hits: int = 2, max_anti_hits: int = 0) -> Tuple[bool, int, int]:
    """
    Keep job if:
    - title is not hard-excluded (C++/embedded)
    - combined text has at least min_target_hits target signals
    - anti_hits == 0 (strict)
    """
    if hard_exclude_by_title(title):
        combined = f"{title}\n{desc}"
        return False, count_signals(combined, TARGET_AREA_SIGNALS), count_signals(combined, ANTI_SIGNALS)

    combined = f"{title}\n{desc}"

    target_hits = count_signals(combined, TARGET_AREA_SIGNALS)
    anti_hits = count_signals(combined, ANTI_SIGNALS)

    if anti_hits > max_anti_hits:
        return False, target_hits, anti_hits

    return (target_hits >= min_target_hits), target_hits, anti_hits


# ----------------------------
# 5) TF-IDF similarity (text-based)
# ----------------------------

def clean_text_for_tfidf(text: str) -> str:
    t = (text or "").lower()
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s\-\+]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tfidf_similarity(a: str, b: str) -> float:
    a2, b2 = clean_text_for_tfidf(a), clean_text_for_tfidf(b)
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    mat = vect.fit_transform([a2, b2])
    sim = cosine_similarity(mat[0:1], mat[1:2])[0][0]
    return float(sim)


# ----------------------------
# 6) Skills score + confidence penalty
# ----------------------------

def skills_score(job_text: str, cv_text: str) -> Tuple[float, Set[str], Set[str], Set[str]]:
    job_sk = extract_skills(job_text)
    cv_sk = extract_skills(cv_text)

    matched = job_sk & cv_sk
    missing = job_sk - cv_sk

    coverage = (len(matched) / len(job_sk)) if job_sk else 0.0

    # If job_sk is tiny, don’t trust 100%
    confidence = min(1.0, (len(job_sk) / 12.0)) if job_sk else 0.0

    return coverage * confidence, job_sk, matched, missing


# ----------------------------
# 7) Hybrid final scoring
# ----------------------------

def hybrid_match(title: str, job_text: str, cv_text: str,
                 w_skills: float = 0.65, w_tfidf: float = 0.35) -> Dict:
    ok, target_hits, anti_hits = is_target_job(title, job_text)
    if not ok:
        return {
            "keep": False,
            "target_hits": target_hits,
            "anti_hits": anti_hits,
            "match_percent": 0.0,
        }

    combined_job = f"{title}\n{job_text}"

    sk_score, job_sk, matched, missing = skills_score(combined_job, cv_text)
    tf_score = tfidf_similarity(cv_text, combined_job)

    final = (w_skills * sk_score) + (w_tfidf * tf_score)

    return {
        "keep": True,
        "target_hits": target_hits,
        "anti_hits": anti_hits,

        "match_percent": round(final * 100, 1),
        "skills_part_percent": round(sk_score * 100, 1),
        "tfidf_part_percent": round(tf_score * 100, 1),

        "job_skills_count": len(job_sk),
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
    }


# ----------------------------
# 8) IO helpers
# ----------------------------

def read_jobs_from_csv(path: str) -> List[Dict]:
    jobs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("description") and row["description"].strip():
                jobs.append(row)
    return jobs

def read_pdf_text(pdf_path: str) -> str:
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)

def read_docx_text(docx_path: str) -> str:
    return docx2txt.process(docx_path) or ""


# ----------------------------
# 9) Main
# ----------------------------

if __name__ == "__main__":
    # CV: choose one
    cv_text = read_pdf_text("yuliia_data_.pdf")
    # cv_text = read_docx_text("yuliia_data_.docx")

    jobs = read_jobs_from_csv("stepstone_jobs_with_description.csv")

    results = []
    skipped = 0

    for i, job in enumerate(jobs, start=1):
        title = job.get("title", "") or ""
        desc = job.get("description", "") or ""
        if not desc.strip() and not title.strip():
            continue

        s = hybrid_match(title, desc, cv_text)

        if not s.get("keep"):
            skipped += 1
            continue

        results.append({
            "title": title,
            "company": job.get("company"),
            "location": job.get("location"),
            "url": job.get("url"),

            "match_percent": s["match_percent"],
            "skills_part_percent": s["skills_part_percent"],
            "tfidf_part_percent": s["tfidf_part_percent"],

            "target_hits": s["target_hits"],
            "anti_hits": s["anti_hits"],
            "job_skills_count": s["job_skills_count"],

            "matched_skills": ", ".join(s["matched_skills"]),
            "missing_skills": ", ".join(s["missing_skills"]),
        })

        print(f"[{i}] {title} → {s['match_percent']}% "
              f"(skills={s['skills_part_percent']}%, tfidf={s['tfidf_part_percent']}%)")

    results.sort(key=lambda x: x["match_percent"], reverse=True)

    out_path = "cv_job_match_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "title", "company", "location", "url",
            "match_percent",
            "skills_part_percent", "tfidf_part_percent",
            "target_hits", "anti_hits",
            "job_skills_count",
            "matched_skills", "missing_skills",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\nSkipped (excluded) jobs: {skipped}")
    print(f"Saved: {out_path}")
