import csv, json, os, re, random
import subprocess

from pathlib import Path
from typing import List, Set

RESULTS_CSV = "cv_job_match_results.csv"
MASTER_CV_MD = "master_cv.md"
TRUE_SKILLS_JSON = "my_true_skills.json"
BULLET_BANK_JSON = "bullet_bank.json"
OUT_DIR = "generated_cv"

BULLETS_TO_INSERT = 4
MAX_SKILLS_IN_SECTION = 40

# главное: весь missing в skills
INCLUDE_ALL_MISSING_IN_SKILLS = True

# SELF_EMPLOYED_HEADER_RE = re.compile(r"^Data Engineer\.\s*Self employed\..*$", flags=re.M)
SELF_EMPLOYED_HEADER_RE = re.compile(
    r"^\s*(?:\*\*)?Data Engineer\.(?:\*\*)?\s*Self employed\.\s*.*$",
    flags=re.M
)

HARD_EXCLUDE_TITLE_PATTERNS = [
    r"\bc\+\+\b", r"\bcpp\b", r"\bqt\b",
    r"\bembedded\b", r"\bfirmware\b", r"\bdriver\b", r"\bkernel\b",
    r"\bdevice driver\b", r"\bmicrocontroller\b", r"\brtos\b",
]

# def md_to_pdf(md_path: Path, pdf_path: Path):
#     cmd = [
#         "markdown-pdf",
#         str(md_path),
#         "-o",
#         str(pdf_path)
#     ]
#     try:
#         subprocess.run(cmd, check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"❌ PDF generation failed for {md_path.name}: {e}")
def md_to_pdf(md_path: Path, pdf_path: Path):
    cmd = ["markdown-pdf", str(md_path), "-o", str(pdf_path), "--css-path", "cv.css"]
    subprocess.run(cmd, check=True)

def canon(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s

SECTION_SKILLS_RE = re.compile(r"^(##\s*)?SKILLS\s*$", re.I)
SECTION_EXPERIENCE_RE = re.compile(r"^(##\s*)?EXPERIENCE\s*$", re.I)

def categorize_skills(skills):
    core, data_cloud, devops, ml, other = set(), set(), set(), set(), set()

    for s in skills:
        x = canon(s)

        if x in {"python", "sql", "scala", "pyspark", "spark", "ray"}:
            core.add(x)
        elif x in {
            "aws", "s3", "ec2", "emr", "athena", "redshift", "lambda", "glue",
            "aws glue", "step functions", "databricks", "delta", "iceberg",
            "lakehouse", "snowflake", "bigquery", "postgres", "postgresql",
            "opensearch", "elasticsearch", "airflow", "dbt", "etl", "elt"
        }:
            data_cloud.add(x)
        elif x in {
            "ci/cd", "github actions", "jenkins",
            "docker", "kubernetes", "terraform",
            "grafana", "prometheus"
        }:
            devops.add(x)
        elif x in {"mlops", "mlflow", "kubeflow", "sagemaker"}:
            ml.add(x)
        else:
            other.add(x)

    return {
        "core": sorted(core),
        "data_cloud": sorted(data_cloud),
        "devops": sorted(devops),
        "ml": sorted(ml),
        "other": sorted(other),
    }


def extract_tools_line_from_master(master_cv: str):
    for ln in master_cv.splitlines():
        if canon(ln).startswith("tools:"):
            return ln.strip()
    return None


def build_skills_block_from_missing(master_cv, missing_skills, max_keywords=35):
    cats = categorize_skills(missing_skills)
    lines = ["SKILLS"]

    if cats["core"]:
        lines.append("Core: " + ", ".join(cats["core"]))
    if cats["data_cloud"]:
        lines.append("Data/Cloud: " + ", ".join(cats["data_cloud"]))
    if cats["devops"]:
        lines.append("DevOps: " + ", ".join(cats["devops"]))
    if cats["ml"]:
        lines.append("ML/MLOps: " + ", ".join(cats["ml"]))

    tools_line = extract_tools_line_from_master(master_cv)
    if tools_line:
        lines.append(tools_line)

    uniq, seen = [], set()
    for s in missing_skills:
        s2 = canon(s)
        if s2 and s2 not in seen:
            uniq.append(s2)
            seen.add(s2)

    uniq = uniq[:max_keywords]
    if uniq:
        lines.append("Keywords: " + ", ".join(uniq))

    lines.append("")
    return lines


def replace_skills_section(cv_text, new_skills_block_lines):
    lines = cv_text.splitlines()

    start = None
    for i, ln in enumerate(lines):
        if SECTION_SKILLS_RE.match(ln.strip()):
            start = i
            break

    if start is None:
        return cv_text

    end = None
    for j in range(start + 1, len(lines)):
        if SECTION_EXPERIENCE_RE.match(lines[j].strip()):
            end = j
            break
    if end is None:
        end = len(lines)

    new_lines = lines[:start] + new_skills_block_lines + lines[end:]
    return "\n".join(new_lines) + "\n"


def parse_list(cell: str) -> List[str]:
    if not cell:
        return []
    return [canon(x) for x in cell.split(",") if canon(x)]

def slugify(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:90] or "job"

def hard_exclude_by_title(title: str) -> bool:
    t = (title or "").lower()
    return any(re.search(p, t) for p in HARD_EXCLUDE_TITLE_PATTERNS)

def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_master_cv() -> str:
    return Path(MASTER_CV_MD).read_text(encoding="utf-8")

def load_true_skills() -> Set[str]:
    data = load_json(TRUE_SKILLS_JSON)
    return {canon(x) for x in data.get("skills", [])}

def load_bullet_bank() -> dict:
    return load_json(BULLET_BANK_JSON)

def pick_4_bullets(priority_skills_for_bullets: List[str], true_skills: Set[str], bullet_bank: dict) -> List[str]:
    tech_bank = bullet_bank.get("tech_signals", {})
    picked = []

    # pick based on skills that are both in true_skills and in bank
    for s in priority_skills_for_bullets:
        if len(picked) >= BULLETS_TO_INSERT:
            break
        if s in true_skills and s in tech_bank and tech_bank[s]:
            b = random.choice(tech_bank[s]).rstrip(".;")
            if b not in picked:
                picked.append(b)

    fallback = ["spark", "airflow", "etl", "sql", "glue", "python", "kafka", "dbt", "terraform", "kubernetes"]
    for s in fallback:
        if len(picked) >= BULLETS_TO_INSERT:
            break
        if s in true_skills and s in tech_bank and tech_bank[s]:
            b = random.choice(tech_bank[s]).rstrip(".;")
            if b not in picked:
                picked.append(b)

    picked = picked[:BULLETS_TO_INSERT]
    out = []
    for i, b in enumerate(picked):
        # out.append(b + (";" if i < BULLETS_TO_INSERT - 1 else "."))
        out.append("• " + b.rstrip(".;") + (";" if i < BULLETS_TO_INSERT - 1 else "."))

    return out

def replace_or_insert_4_lines_under_self_employed(cv_text: str, new_lines: List[str]) -> str:
    m = SELF_EMPLOYED_HEADER_RE.search(cv_text)
    if not m:
        raise RuntimeError("❌ Self-employed header not found in master_cv.md")

    lines = cv_text.splitlines()

    header_idx = None
    for i, ln in enumerate(lines):
        if canon(ln) == canon(m.group(0)):
            header_idx = i
            break
    if header_idx is None:
        for i, ln in enumerate(lines):
            if "data engineer." in canon(ln) and "self employed." in canon(ln):
                header_idx = i
                break
    if header_idx is None:
        raise RuntimeError("❌ Could not locate header line index")

    months = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    next_header_re = re.compile(
        rf"^[A-Za-z ].+\.\s*[A-Za-z0-9&()' -]+\.\s*{months}\s+\d{{4}}\s*-\s*(?:{months}\s+\d{{4}}|Present)$"
    )

    start = header_idx + 1
    end = len(lines)
    for j in range(start, len(lines)):
        if next_header_re.match(lines[j].strip()):
            end = j
            break

    block = lines[start:end]
    non_empty = [k for k, ln in enumerate(block) if ln.strip()]

    replace_count = min(len(non_empty), BULLETS_TO_INSERT)
    for r in range(replace_count):
        block[non_empty[r]] = new_lines[r]

    if replace_count < BULLETS_TO_INSERT:
        to_add = new_lines[replace_count:BULLETS_TO_INSERT]
        if block and block[-1].strip() != "":
            block.append("")
        block.extend(to_add)

    new_all = lines[:start] + block + lines[end:]
    return "\n".join(new_all) + "\n"

def replace_or_insert_4_lines_under_self_employed_plain(cv_text: str, new_lines: List[str]) -> str:
    """
    Same as replace_or_insert_4_lines_under_self_employed, but:
    - does NOT insert extra blank lines
    - keeps plain lines style
    """
    m = SELF_EMPLOYED_HEADER_RE.search(cv_text)
    if not m:
        raise RuntimeError("❌ Self-employed header not found in master_cv.md")

    lines = cv_text.splitlines()

    # robust header index: match by regex, not exact string equality
    header_idx = None
    for i, ln in enumerate(lines):
        if SELF_EMPLOYED_HEADER_RE.match(ln):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("❌ Could not locate header line index")

    months = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    next_header_re = re.compile(
        rf"^\s*(?:\*\*)?.+\.(?:\*\*)?\s*[A-Za-z0-9&()' -]+\.\s*{months}\s+\d{{4}}\s*-\s*(?:{months}\s+\d{{4}}|Present)\s*$"
    )

    start = header_idx + 1
    end = len(lines)
    for j in range(start, len(lines)):
        if next_header_re.match(lines[j].strip()):
            end = j
            break

    block = lines[start:end]
    non_empty = [k for k, ln in enumerate(block) if ln.strip()]

    # replace first 4 non-empty lines
    replace_count = min(len(non_empty), BULLETS_TO_INSERT)
    for r in range(replace_count):
        block[non_empty[r]] = new_lines[r]

    # append remaining without adding blank lines
    if replace_count < BULLETS_TO_INSERT:
        to_add = [x for x in new_lines[replace_count:BULLETS_TO_INSERT] if x.strip()]
        block.extend(to_add)

    new_all = lines[:start] + block + lines[end:]
    return "\n".join(new_all) + "\n"

def upsert_skills_section(cv_text: str, prioritized_skills: List[str]) -> str:
    prioritized_skills = [canon(x) for x in prioritized_skills if x.strip()]

    lines = cv_text.splitlines()

    # ищем SKILLS / Skills / ## Skills
    header_idx = None
    for i, ln in enumerate(lines):
        if canon(ln) in ("skills", "## skills", "# skills"):
            header_idx = i
            break

    def normalize_skill_line(ln: str):
        return canon(re.sub(r"^[\-•]\s*", "", ln))

    # если секция есть
    if header_idx is not None:
        j = header_idx + 1
        existing = []
        while j < len(lines):
            ln = lines[j]
            if not ln.strip():
                break
            if re.match(r"^[A-Z][A-Z\s]{2,}$", ln):  # новый раздел
                break
            existing.append(normalize_skill_line(ln))
            j += 1

        merged = []
        for s in prioritized_skills + existing:
            if s and s not in merged:
                merged.append(s)

        # new_block = ["SKILLS"] + [f"- {s}" for s in merged] + [""]
        skills_line = ", ".join(merged[:MAX_SKILLS_IN_SECTION])
        new_block = ["SKILLS", skills_line, ""]

        return "\n".join(lines[:header_idx] + new_block + lines[j:]) + "\n"

    # если Skills нет — вставляем после OBJECTIVE
    for i, ln in enumerate(lines):
        if canon(ln) == "objective":
            insert_at = i + 2
            # plain lines (без маркеров)

            skills_line = ", ".join(prioritized_skills[:MAX_SKILLS_IN_SECTION])
            new_block = ["SKILLS", skills_line, ""]

            # new_block = ["SKILLS"] + [f"{s}" for s in prioritized_skills] + [""]
            #
            # new_block = ["SKILLS"] + [f"- {s}" for s in prioritized_skills] + [""]
            return "\n".join(lines[:insert_at] + new_block + lines[insert_at:]) + "\n"

    return cv_text

def main():
    random.seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)

    master_cv = load_master_cv()
    true_skills = load_true_skills()
    bullet_bank = load_bullet_bank()

    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    generated = 0
    skipped = 0

    for row in rows:
        title = row.get("title", "") or ""
        company = row.get("company", "") or ""
        url = row.get("url", "") or ""

        if hard_exclude_by_title(title):
            skipped += 1
            continue

        missing = parse_list(row.get("missing_skills", ""))
        matched = parse_list(row.get("matched_skills", ""))

        # Skills block uses ALL missing skills for ATS + pretty categories
        if not missing and not matched:
            skipped += 1
            continue

        # Bullets: ONLY from true skills (honest)
        priority_for_bullets = [s for s in missing if s in true_skills]
        if not priority_for_bullets:
            priority_for_bullets = [s for s in matched if s in true_skills]

        new_4 = pick_4_bullets(priority_for_bullets, true_skills, bullet_bank)

        # 1) replace self-employed block
        try:
            tailored = replace_or_insert_4_lines_under_self_employed_plain(master_cv, new_4)
        except RuntimeError as e:
            print(f"Skip '{title}' — {e}")
            skipped += 1
            continue

        # 2) pretty SKILLS block from missing skills
        skills_block = build_skills_block_from_missing(master_cv, missing, max_keywords=35)
        tailored = replace_skills_section(tailored, skills_block)

        # 3) save MD + PDF
        fname = f"{slugify(company)}__{slugify(title)}.md"
        md_path = Path(OUT_DIR) / fname
        pdf_path = md_path.with_suffix(".pdf")

        md_path.write_text(tailored, encoding="utf-8")
        md_to_pdf(md_path, pdf_path)

        generated += 1

    print(f"✅ Done. Generated CVs: {generated}. Skipped: {skipped}. Output: {OUT_DIR}/")

if __name__ == "__main__":
    main()
