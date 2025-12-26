import json
import re
import time
import html as html_lib
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.stepstone.de"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
    "Referer": "https://www.stepstone.de/",
}

# Ищем начало JSON-объекта вакансии
JOB_START_RE = re.compile(
    r'\{"id":\d+,"title":"[^"]+","labels":\[\],"url":"/stellenangebote',
    re.MULTILINE
)


def extract_braced_object(text: str, start_idx: int) -> str | None:
    """Вырезает JSON объект начиная с '{' по балансу скобок, учитывая строки/экранирование."""
    n = len(text)
    i = start_idx

    if i >= n or text[i] != "{":
        return None

    depth = 0
    in_str = False
    esc = False
    begin = i

    while i < n:
        ch = text[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[begin: i + 1]
        i += 1

    return None


def extract_jobs_from_html_payload(html: str):
    jobs = []

    for m in JOB_START_RE.finditer(html):
        obj_str = extract_braced_object(html, m.start())
        if not obj_str:
            continue

        try:
            j = json.loads(obj_str)
        except json.JSONDecodeError:
            continue

        url = urljoin(BASE_URL, j.get("url", ""))
        if "/stellenangebote" not in url:
            continue

        jobs.append(
            {
                "id": j.get("id"),
                "title": j.get("title"),
                "company": j.get("companyName"),
                "location": j.get("location"),
                "datePosted": j.get("datePosted"),
                "url": url,
            }
        )

    uniq = {x["url"]: x for x in jobs if x.get("url")}
    return list(uniq.values())


def fetch_jobs(search_url: str, pages: int = 3):
    session = requests.Session()
    session.headers.update(HEADERS)

    all_jobs = []

    for page in range(1, pages + 1):
        url = search_url + (("&" if "?" in search_url else "?") + f"page={page}")
        print(f"Fetching: {url}")

        r = session.get(url, timeout=30)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else None

        jobs = extract_jobs_from_html_payload(r.text)

        print(f"  title={title}, html_len={len(r.text)}, jobs_on_page={len(jobs)}")

        for j in jobs:
            j["page"] = page
        all_jobs.extend(jobs)

    uniq = {j["url"]: j for j in all_jobs}
    return list(uniq.values())


# -------------------------
# Job Description extraction
# -------------------------

def _clean_text(txt: str) -> str:
    txt = html_lib.unescape(txt or "")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def extract_description_from_job_page(html: str) -> str | None:
    """
    Stepstone job page:
    1) Берём description из JSON-LD JobPosting (самый чистый способ).
    2) Если JSON-LD сломан/нестандартный — пробуем вытащить "description":"..." regex'ом.
    3) Последний fallback: main/article.
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1) JSON-LD JobPosting
    for s in soup.select('script[type="application/ld+json"]'):
        raw = (s.string or s.get_text() or "").strip()
        if not raw:
            continue

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None

        if data is not None:
            candidates = data if isinstance(data, list) else [data]
            for item in candidates:
                if not isinstance(item, dict):
                    continue

                # JobPosting прямо
                if item.get("@type") == "JobPosting" and item.get("description"):
                    desc_html = item["description"]
                    desc_text = BeautifulSoup(desc_html, "html.parser").get_text(" ", strip=True)
                    return _clean_text(desc_text)

                # JobPosting внутри @graph
                graph = item.get("@graph")
                if isinstance(graph, list):
                    for g in graph:
                        if isinstance(g, dict) and g.get("@type") == "JobPosting" and g.get("description"):
                            desc_html = g["description"]
                            desc_text = BeautifulSoup(desc_html, "html.parser").get_text(" ", strip=True)
                            return _clean_text(desc_text)

        # 2) fallback: если JSON-LD не распарсился, но внутри есть "description":"...."
        if '"@type":"JobPosting"' in raw and '"description"' in raw:
            m = re.search(r'"description"\s*:\s*"((?:\\.|[^"\\])*)"', raw, flags=re.DOTALL)
            if m:
                # это JSON-escaped строка
                desc_escaped = m.group(1)
                desc_html = bytes(desc_escaped, "utf-8").decode("unicode_escape")
                desc_text = BeautifulSoup(desc_html, "html.parser").get_text(" ", strip=True)
                return _clean_text(desc_text)

    # 3) fallback по видимому тексту (грубее)
    main = soup.find("main")
    if main:
        text = _clean_text(main.get_text(" ", strip=True))
        if len(text) > 300:
            return text

    article = soup.find("article")
    if article:
        text = _clean_text(article.get_text(" ", strip=True))
        if len(text) > 300:
            return text

    return None



def enrich_jobs_with_description(jobs, max_jobs: int = 50, sleep_s: float = 1.0):
    """
    Для первых max_jobs вакансий подтягиваем описание.
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    out = []
    for i, j in enumerate(jobs[:max_jobs], start=1):
        url = j["url"]
        print(f"  [{i}/{min(max_jobs, len(jobs))}] Fetching JD: {url}")

        r = session.get(url, timeout=30)
        r.raise_for_status()

        desc = extract_description_from_job_page(r.text)
        j2 = dict(j)
        j2["description"] = desc
        out.append(j2)

        time.sleep(sleep_s)  # чтобы не спамить запросами

    # остальные без описания (если нужно сохранить весь список)
    for j in jobs[max_jobs:]:
        j2 = dict(j)
        j2["description"] = None
        out.append(j2)

    return out

import csv

def save_jobs_to_csv(jobs, path="stepstone_jobs.csv"):
    fields = ["title", "company", "location", "datePosted", "url", "description"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for j in jobs:
            w.writerow({k: j.get(k) for k in fields})


if __name__ == "__main__":
    SEARCH_URL = "https://www.stepstone.de/jobs/data-engineer/in-nordrhein-westfalen?whereType=autosuggest&radius=30&action=facet_selected%3bdetectedLanguages%3ben&fdl=en&searchOrigin=Resultlist_top-search"
    jobs = fetch_jobs(SEARCH_URL, pages=5)
    jobs = enrich_jobs_with_description(jobs, max_jobs=50, sleep_s=1.0)
    save_jobs_to_csv(jobs, "stepstone_jobs_with_description.csv")
    print("Saved: stepstone_jobs_with_description.csv")

