import json
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from config import PROCESSED_DIR, INTERIM_DIR
from utils import print_header, save_csv, ensure_dir

load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing in .env")
    return OpenAI(api_key=api_key)


def get_search_language(country: str) -> str:
    mapping = {
        "France": "French",
        "Estonia": "Estonian",
        "Germany": "German",
        "Austria": "German",
        "Belgium": "Dutch and French",
        "Bulgaria": "Bulgarian",
        "Cyprus": "Greek and English",
        "Italy": "Italian",
        "Spain": "Spanish",
        "Portugal": "Portuguese",
        "Netherlands": "Dutch",
        "Finland": "Finnish and Swedish",
        "Sweden": "Swedish",
        "Denmark": "Danish",
        "Norway": "Norwegian",
        "Poland": "Polish",
        "Romania": "Romanian",
        "Croatia": "Croatian",
        "Lithuania": "Lithuanian",
        "Latvia": "Latvian",
        "Luxembourg": "French and German",
        "Malta": "English and Maltese",
        "Slovakia": "Slovak",
        "Slovenia": "Slovenian",
        "Czechia": "Czech",
        "Hungary": "Hungarian",
        "Greece": "Greek",
        "Switzerland": "German, French, or Italian",
        "Albania": "Albanian",
        "Bosnia and Herzegovina": "Bosnian",
        "Montenegro": "Montenegrin",
        "North Macedonia": "Macedonian",
        "Serbia": "Serbian",
        "Ukraine": "Ukrainian",
        "Iceland": "Icelandic",
        "Ireland": "English",
    }
    return mapping.get(country, "English")


def get_language_instruction(country: str) -> str:
    mapping = {
        "France": "Generate the queries mainly in French.",
        "Estonia": "Generate the queries mainly in Estonian, but include English where useful for official documents.",
        "Germany": "Generate the queries mainly in German.",
        "Austria": "Generate the queries mainly in German.",
        "Belgium": "Generate a mix of Dutch and French queries where appropriate.",
        "Bulgaria": "Generate the queries mainly in Bulgarian.",
        "Cyprus": "Generate the queries mainly in Greek, with English where useful for official documents.",
        "Italy": "Generate the queries mainly in Italian.",
        "Spain": "Generate the queries mainly in Spanish.",
        "Portugal": "Generate the queries mainly in Portuguese.",
        "Netherlands": "Generate the queries mainly in Dutch.",
        "Finland": "Generate the queries mainly in Finnish, with Swedish or English where appropriate.",
        "Sweden": "Generate the queries mainly in Swedish.",
        "Denmark": "Generate the queries mainly in Danish.",
        "Norway": "Generate the queries mainly in Norwegian.",
        "Poland": "Generate the queries mainly in Polish.",
        "Romania": "Generate the queries mainly in Romanian.",
        "Croatia": "Generate the queries mainly in Croatian.",
        "Lithuania": "Generate the queries mainly in Lithuanian.",
        "Latvia": "Generate the queries mainly in Latvian.",
        "Luxembourg": "Generate the queries mainly in French or German where appropriate.",
        "Malta": "Generate the queries mainly in English, with Maltese where useful.",
        "Slovakia": "Generate the queries mainly in Slovak.",
        "Slovenia": "Generate the queries mainly in Slovenian.",
        "Czechia": "Generate the queries mainly in Czech.",
        "Hungary": "Generate the queries mainly in Hungarian.",
        "Greece": "Generate the queries mainly in Greek.",
        "Switzerland": "Generate the queries mainly in German, French, or Italian depending on likely official source usage.",
        "Albania": "Generate the queries mainly in Albanian.",
        "Bosnia and Herzegovina": "Generate the queries mainly in Bosnian.",
        "Montenegro": "Generate the queries mainly in Montenegrin.",
        "North Macedonia": "Generate the queries mainly in Macedonian.",
        "Serbia": "Generate the queries mainly in Serbian.",
        "Ukraine": "Generate the queries mainly in Ukrainian.",
        "Iceland": "Generate the queries mainly in Icelandic.",
        "Ireland": "Generate the queries in English.",
    }
    return mapping.get(country, "Generate the queries in English.")


def parse_json_output(text: str) -> dict:
    text = safe_text(text)

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    return {}


def strip_explanation_block(question_text: str) -> str:
    text = safe_text(question_text)
    text = re.split(r"(?i)for explanation\s*:", text, maxsplit=1)[0].strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def shorten_query(text: str, max_len: int = 220) -> str:
    text = safe_text(text)
    if len(text) <= max_len:
        return text

    cut = text[:max_len]
    last_space = cut.rfind(" ")
    if last_space > 80:
        cut = cut[:last_space]

    return cut.strip()


def clean_and_limit_queries(queries: list[str], max_queries: int = 5, max_len: int = 300) -> list[str]:
    cleaned = []
    seen = set()

    for q in queries:
        q = shorten_query(safe_text(q), max_len=max_len)
        if not q:
            continue
        if q.lower() in seen:
            continue
        seen.add(q.lower())
        cleaned.append(q)

    return cleaned[:max_queries]


def get_question_search_hint(question_id: str, dimension: str) -> str:
    qid = safe_text(question_id)

    hints = {
        "P1": "Prioritise laws, legal acts, decrees, official legal databases, directive transposition documents, and national policy framework documents.",
        "P2": "Prioritise national open data strategy, roadmap, implementation strategy, official strategic documents, and ministry policy pages.",
        "P3": "Prioritise municipal, city, regional, or local government open data strategy and policy sources rather than national-only documents.",
        "P4": "Prioritise action plans, implementation plans, roadmaps, milestones, programme delivery plans, and operational policy documents.",
        "P13": "Prioritise governance structures, official responsible ministry or agency pages, steering groups, coordination bodies, and public-sector governance documents.",
        "P16": "Prioritise evidence showing breadth across regions, municipalities, or public bodies. Queries should target extent and spread, not just one isolated example.",
        "P22": "Prioritise data publication plans, release schedules, publication guidance, obligations, and official instructions for public bodies.",
        "P24": "Prioritise policy update cycles, revised strategy documents, review procedures, update announcements, and official revision documents.",
        "I1": "Prioritise legal definitions, law text, official definitions of reuse, public information law, public sector information reuse, and legal database pages.",
        "I2": "Prioritise monitoring methodologies, dashboards, annual assessment reports, measurement frameworks, indicators, reuse monitoring processes, and official reporting pages.",
        "I9": "Prioritise consultations, workshops, stakeholder engagement, user-needs assessments, reuse stimulation events, and official participation records.",
        "I10": "Prioritise official reuse case collections, showcase pages, examples of open data reuse, public sector information reuse cases, yearly collections of reuse cases, and government case-study pages.",
        "I12": "Prioritise government impact reports, efficiency reports, transparency impact, public service improvement evidence, and official case-based evidence of open data or PSI reuse.",
        "I17": "Prioritise social impact reports, inclusion, education, awareness, public-value outcomes, case studies, and official evidence of social outcomes from open data or PSI reuse.",
        "I27": "Prioritise economic impact reports, innovation, startup or business reuse, value creation studies, economic assessments, and official evidence of economic outcomes from open data or PSI reuse.",
    }

    if qid in hints:
        return hints[qid]

    if safe_text(dimension).lower() == "policy":
        return "Prioritise laws, strategies, decrees, governance structures, official ministry pages, and implementation documents."
    return "Prioritise monitoring, methodologies, reports, case studies, examples of open data or PSI reuse, and documented impact evidence."


def get_question_disambiguation_hint(question_id: str, dimension: str) -> str:
    qid = safe_text(question_id)

    hints = {
        "I1": (
            "This question is about the legal or policy definition of reuse of open data or public sector information. "
            "Do not drift toward unrelated uses of the word reuse."
        ),
        "I2": (
            "This question is about monitoring reuse of open data or public sector information. "
            "Do not drift toward generic monitoring, cybersecurity monitoring, or unrelated operational monitoring."
        ),
        "I10": (
            "This question is about reuse cases of open data or public sector information. "
            "Do not drift toward recycling, material reuse, waste reuse, or non-data reuse."
        ),
        "I12": (
            "This question is about governmental impact of open data or PSI reuse. "
            "Do not drift toward broad digital policy promises unless they specifically document impact."
        ),
        "I17": (
            "This question is about social impact of open data or PSI reuse. "
            "Do not drift toward generic social policy or unrelated community projects unless open data reuse is explicit."
        ),
        "I27": (
            "This question is about economic impact of open data or PSI reuse. "
            "Do not drift toward general digital economy claims unless they are tied to open data or public-sector-information reuse."
        ),
    }

    return hints.get(qid, "")


def get_country_source_hint(country: str, question_id: str) -> str:
    country = safe_text(country)
    question_id = safe_text(question_id)

    general = "Prefer official government, ministry, portal, legal database, statistical office, or public-sector sources."

    country_hints = {
        "Estonia": {
            "default": (
                "For Estonia, prefer official sources such as legal databases, ministry pages, official open data portals, "
                "public-sector policy or methodology documents, and authoritative report pages. Avoid drifting toward e-Residency, "
                "procurement, transparency NGO pages, broad unrelated digital-government pages, or generic programme annexes unless they directly answer the question."
            ),
            "I1": (
                "For Estonia I1, strongly prefer legal-definition sources such as the Public Information Act, official legal database pages, "
                "and official wording about reuse of public information or reuse of public sector information."
            ),
            "I2": (
                "For Estonia I2, strongly prefer monitoring, methodology, annual assessment, indicators, reuse measurement, "
                "dashboard, or official reporting sources specifically about open data or public information reuse."
            ),
            "I10": (
                "For Estonia I10, strongly prefer official examples, reuse case collections, showcase pages, annual collection exercises, "
                "or official documentation of open data or PSI reuse cases. Avoid recycling or waste reuse."
            ),
            "I12": (
                "For Estonia I12, strongly prefer official reports or evidence of governmental impact, efficiency, better services, "
                "administrative improvement, or transparency outcomes caused by open data or PSI reuse."
            ),
            "I17": (
                "For Estonia I17, strongly prefer social impact evidence, inclusion, education, awareness, or public-value outcomes from open data or PSI reuse."
            ),
            "I27": (
                "For Estonia I27, strongly prefer official economic impact, value creation, innovation, startup or business reuse, "
                "or assessment reports tied to open data or public information reuse."
            ),
            "P2": (
                "For Estonia P2, strongly prefer official national strategy or roadmap sources specifically about open data rather than generic digital government."
            ),
            "P4": (
                "For Estonia P4, strongly prefer action plan, roadmap, implementation plan, milestones, or operational policy documents."
            ),
        }
    }

    if country in country_hints:
        per_country = country_hints[country]
        return per_country.get(question_id, per_country.get("default", general))

    return general


def get_country_query_keywords(country: str, question_id: str) -> list[str]:
    country = safe_text(country)
    question_id = safe_text(question_id)

    if country == "Estonia":
        mapping = {
            "I1": ["avaliku teabe seadus", "taaskasutus", "public information reuse", "public sector information reuse", "riigiteataja"],
            "I2": ["avaandmete taaskasutuse seire", "reuse monitoring", "annual assessment", "metoodika", "indikatorid"],
            "I10": ["avaandmete taaskasutuse näited", "open data reuse cases", "public sector information reuse examples", "showcase", "case study"],
            "I12": ["avaandmete mõju", "government impact of open data", "public sector information reuse impact", "impact assessment"],
            "I17": ["open data social impact", "public value", "social outcomes", "avaandmete sotsiaalne mõju"],
            "I27": ["open data economic impact", "business reuse", "innovation", "avaandmete majanduslik mõju"],
            "P2": ["avaandmete strateegia", "open data strategy", "roadmap"],
            "P4": ["tegevuskava", "implementation plan", "roadmap", "milestones"],
        }
        return mapping.get(question_id, [])

    return []


def get_negative_keyword_hint(question_id: str) -> str:
    qid = safe_text(question_id)

    hints = {
        "I10": "Avoid queries that could match recycling, waste reuse, circular economy, or material reuse.",
        "I12": "Avoid queries that drift into general data policy without explicit open data or PSI reuse impact.",
        "I17": "Avoid generic social policy wording unless open data or PSI reuse is explicit.",
        "I27": "Avoid generic digital economy claims unless open data or PSI reuse is explicit.",
        "I2": "Avoid generic operational or security monitoring unrelated to open data or PSI reuse.",
    }
    return hints.get(qid, "")


def fallback_queries(country: str, question: str, dimension: str, question_id: str = "") -> list[str]:
    short_question = strip_explanation_block(question)
    short_question = shorten_query(short_question, max_len=220)
    qid = safe_text(question_id)

    extra_keywords = get_country_query_keywords(country, qid)
    extra_text = " ".join(extra_keywords).strip()

    if qid == "I10":
        queries = [
            f'{country} "{short_question}" "open data reuse"',
            f'{country} "public sector information reuse" showcase examples {extra_text}'.strip(),
            f'{country} "open data reuse cases" official government {extra_text}'.strip(),
            f'{country} "public information reuse examples" official {extra_text}'.strip(),
            f'{country} "open data case study" government {extra_text}'.strip(),
        ]
    elif qid == "I1":
        queries = [
            f'{country} "{short_question}" law definition',
            f'{country} "public information reuse" legal definition {extra_text}'.strip(),
            f'{country} "public sector information reuse" law {extra_text}'.strip(),
            f'{country} "open data reuse definition" official law {extra_text}'.strip(),
            f'{country} legal database reuse of public information {extra_text}'.strip(),
        ]
    elif qid in {"I12", "I17", "I27"}:
        queries = [
            f'{country} "{short_question}" "open data"',
            f'{country} "open data reuse impact" official report {extra_text}'.strip(),
            f'{country} "public sector information reuse impact" report {extra_text}'.strip(),
            f'{country} "open data impact assessment" government {extra_text}'.strip(),
            f'{country} "open data case study" official report {extra_text}'.strip(),
        ]
    elif safe_text(dimension).lower() == "policy":
        queries = [
            f'{country} "{short_question}"',
            f"{country} open data policy strategy law {extra_text}".strip(),
            f"{country} official open data government document {extra_text}".strip(),
            f"{country} open data strategy pdf {extra_text}".strip(),
            f"{country} open data law ministry {extra_text}".strip(),
        ]
    else:
        queries = [
            f'{country} "{short_question}"',
            f"{country} open data impact reuse report {extra_text}".strip(),
            f"{country} open data case study government {extra_text}".strip(),
            f"{country} public sector information reuse impact {extra_text}".strip(),
            f"{country} open data monitoring methodology {extra_text}".strip(),
        ]

    return clean_and_limit_queries(queries, max_queries=5, max_len=300)


def build_query_generation_prompt(country: str, question_id: str, question: str, dimension: str) -> str:
    search_language = get_search_language(country)
    language_instruction = get_language_instruction(country)

    main_question_only = strip_explanation_block(question)
    question_search_hint = get_question_search_hint(question_id, dimension)
    question_disambiguation_hint = get_question_disambiguation_hint(question_id, dimension)
    country_source_hint = get_country_source_hint(country, question_id)
    country_keywords = get_country_query_keywords(country, question_id)
    negative_keyword_hint = get_negative_keyword_hint(question_id)

    keyword_text = ", ".join(country_keywords) if country_keywords else "None"

    prompt = f"""
You are generating web search queries for a Retrieval-Augmented Generation (RAG) system.

Country: {country}
Country search language: {search_language}
Dimension: {dimension}
Question ID: {question_id}
Question: {main_question_only}

Task:
Generate 5 web search queries that are likely to retrieve high-quality evidence for answering this ODM question.

Requirements:
- {language_instruction}
- The queries should be suitable for finding:
  - official government documents
  - laws or decrees
  - policy papers
  - strategies
  - ministry or agency pages
  - official reports
  - methodology or impact reports
- Make the queries concise and realistic for search engines.
- Include the country name naturally.
- Avoid social media, blogs, vague phrases, procurement notices, tender pages, or unrelated commercial pages.
- Prefer document-oriented and official-source wording.
- The queries should help retrieve evidence for this specific question, not just general open data information.
- Keep the search intent focused on open data, public sector information, public information reuse, monitoring, strategy, or impact as relevant.
- Avoid copying the full long question verbatim if it would make the query unnatural.
- Keep each query reasonably short for web search and API compatibility.

Question-specific search guidance:
{question_search_hint}

Question-specific disambiguation guidance:
{question_disambiguation_hint if question_disambiguation_hint else "None"}

Country-specific source guidance:
{country_source_hint}

Helpful country- and question-specific keyword ideas:
{keyword_text}

Negative matching guidance:
{negative_keyword_hint if negative_keyword_hint else "None"}

Return JSON only in this format:
{{
  "queries": [
    "...",
    "...",
    "...",
    "...",
    "..."
  ]
}}
""".strip()

    return prompt


def generate_queries_core(
    client: OpenAI,
    country: str,
    question_id: str,
    question: str,
    dimension: str,
) -> list[str]:
    prompt = build_query_generation_prompt(
        country=country,
        question_id=question_id,
        question=question,
        dimension=dimension,
    )

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
    )

    parsed = parse_json_output(response.output_text)
    queries = parsed.get("queries", [])

    cleaned = clean_and_limit_queries(queries, max_queries=5, max_len=300)

    if len(cleaned) < 4:
        return fallback_queries(country, question, dimension, question_id)

    return cleaned


def generate_queries_for_row(
    client: OpenAI,
    country: str,
    question_id: str,
    question: str,
    dimension: str
) -> list[str]:
    return generate_queries_core(
        client=client,
        country=country,
        question_id=question_id,
        question=question,
        dimension=dimension,
    )


def generate_queries_for_question(
    country: str,
    question_id: str,
    question_text: str,
    dimension: str,
    output_dir: Path = None,
    client: OpenAI = None,
) -> pd.DataFrame:
    if client is None:
        client = get_openai_client()

    queries = generate_queries_core(
        client=client,
        country=country,
        question_id=question_id,
        question=question_text,
        dimension=dimension,
    )

    short_question = strip_explanation_block(question_text)
    short_question = shorten_query(short_question, max_len=220)

    short_dimension = safe_text(dimension)
    short_country = safe_text(country)
    short_qid = safe_text(question_id)

    rows = []
    for rank, query in enumerate(queries, start=1):
        rows.append({
            "country": short_country,
            "question_id": short_qid,
            "dimension": short_dimension,
            "question": short_question,
            "query_rank": rank,
            "query": query,
            "search_language": get_search_language(country),
        })

    out_df = pd.DataFrame(rows)

    if output_dir is not None:
        output_path = output_dir / "queries.csv"
        out_df.to_csv(output_path, index=False)

    return out_df


def main():
    print_header("STEP 3: GENERATE SEARCH QUERIES WITH LLM")

    input_path = PROCESSED_DIR / "ground_truth_odm_selected.csv"
    output_dir = INTERIM_DIR / "queries"
    output_path = output_dir / "generated_queries_odm_selected.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {input_path}\n"
            f"Please run step2_build_ground_truth.py first."
        )

    ensure_dir(output_dir)

    df = pd.read_csv(input_path)
    client = get_openai_client()

    rows = []

    for _, row in df.iterrows():
        country = safe_text(row.get("country", ""))
        question_id = safe_text(row.get("question_id", ""))
        question = safe_text(row.get("question", ""))
        dimension = safe_text(row.get("dimension", ""))

        try:
            queries = generate_queries_for_row(
                client=client,
                country=country,
                question_id=question_id,
                question=question,
                dimension=dimension,
            )
        except Exception as e:
            print(f"Query generation failed for {country} | {question_id}: {e}")
            queries = fallback_queries(country, question, dimension, question_id)

        for i, query in enumerate(queries, start=1):
            rows.append({
                "country": country,
                "question_id": question_id,
                "dimension": dimension,
                "question": strip_explanation_block(question),
                "query_no": i,
                "query": query,
                "search_language": get_search_language(country),
            })

        print(f"Generated queries for {country} | {question_id} | language={get_search_language(country)}")

    out_df = pd.DataFrame(rows)
    save_csv(out_df, output_path)

    print(f"Saved generated queries to: {output_path}")
    print(f"Rows: {len(out_df)}")
    print("\nPreview:")
    print(out_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()