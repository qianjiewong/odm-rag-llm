import ast
import json
import os
import re
from difflib import get_close_matches

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from config import PROCESSED_DIR, GENERATION_DIR, INTERIM_DIR
from utils import print_header, save_csv, ensure_dir

load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_EVIDENCE_CHUNKS = int(os.getenv("MAX_EVIDENCE_CHUNKS", "8"))


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_whitespace(text: str) -> str:
    return " ".join(safe_text(text).split())


def canonicalize_final_answer(answer: str, response_scoring=None) -> str:
    raw = normalize_whitespace(answer)
    if not raw:
        return raw

    candidate_options = []

    if isinstance(response_scoring, dict):
        candidate_options = [normalize_whitespace(k) for k in response_scoring.keys()]
    elif isinstance(response_scoring, list):
        candidate_options = [normalize_whitespace(x) for x in response_scoring]

    raw_lower = raw.lower()

    for option in candidate_options:
        if raw_lower == option.lower():
            return option

    canonical_map = {
        "yes": "Yes",
        "no": "No",
        "partially": "Partially",
        "partial": "Partially",
        "not applicable": "Not applicable",
        "n/a": "Not applicable",
        "unknown": "Unknown",
        "other": "Other",
        "all public bodies": "All public bodies",
        "the majority of public bodies": "The majority of public bodies",
        "a minority of public bodies": "A minority of public bodies",
        "none of the public bodies": "None of the public bodies",
    }

    if raw_lower in canonical_map:
        return canonical_map[raw_lower]

    if len(raw.split()) <= 4:
        return raw[:1].upper() + raw[1:]

    return raw


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing in .env")
    return OpenAI(api_key=api_key)


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


def parse_response_scoring(raw: str) -> dict:
    raw = safe_text(raw)
    if not raw:
        return {}

    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            cleaned = {}
            for k, v in parsed.items():
                key = safe_text(k)
                try:
                    cleaned[key] = float(v)
                except Exception:
                    cleaned[key] = v
            return cleaned
    except Exception:
        pass

    return {}


def response_option_list(score_map: dict) -> list[str]:
    return [safe_text(k) for k in score_map.keys() if safe_text(k)]


def format_score_map(score_map: dict) -> str:
    if not score_map:
        return "No official response scoring map available."
    lines = []
    for k, v in score_map.items():
        lines.append(f"- {k} => {v}")
    return "\n".join(lines)


def normalize_response_text(text: str) -> str:
    text = safe_text(text).lower()
    text = re.sub(r"[^a-z0-9\s/_-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    synonym_map = {
        "yes partly": "partially",
        "partial yes": "partially",
        "partially yes": "partially",
        "yes partially": "partially",
        "to some extent": "partially",
        "somewhat": "partially",
        "limited yes": "partially",
        "mostly yes": "yes",
        "fully yes": "yes",
        "not really": "no",
        "none": "no",
        "not available": "no",
        "not present": "no",
    }

    return synonym_map.get(text, text)


def validate_selected_response(selected_response: str, response_options: list[str]) -> str:
    selected_response = normalize_response_text(selected_response)

    if not response_options:
        return safe_text(selected_response)

    normalized_option_map = {normalize_response_text(opt): opt for opt in response_options}

    if selected_response in normalized_option_map:
        return normalized_option_map[selected_response]

    matches = get_close_matches(selected_response, list(normalized_option_map.keys()), n=1, cutoff=0.75)
    if matches:
        return normalized_option_map[matches[0]]

    return safe_text(selected_response)


def infer_score_from_selected_response(selected_response: str, score_map: dict):
    if not score_map:
        return None

    normalized_selected = normalize_response_text(selected_response)

    for opt, score in score_map.items():
        if normalized_selected == normalize_response_text(opt):
            return score

    return None


def build_evidence_block(question_df: pd.DataFrame) -> str:
    blocks = []
    used = question_df.head(MAX_EVIDENCE_CHUNKS).copy()

    for i, (_, row) in enumerate(used.iterrows(), start=1):
        blocks.append(
            f"""[S{i}]
Title: {safe_text(row.get("source_title", ""))}
URL: {safe_text(row.get("source_url", ""))}
File type: {safe_text(row.get("source_file_type", ""))}
Similarity score: {row.get("similarity_score", "")}
Chunk text:
{safe_text(row.get("chunk_text", ""))}
"""
        )

    return "\n\n".join(blocks)


def get_question_hint(question_id: str, dimension: str) -> str:
    qid = safe_text(question_id)
    dim = safe_text(dimension).lower()

    hints = {
        "P1": "Look for a formal national open data policy framework, legal basis, and where relevant transposition of the EU Open Data Directive. Prioritise laws, decrees, official strategy pages, and ministry or government documents.",
        "P2": "Look specifically for a national open data strategy, roadmap, or equivalent official strategic document. A broader digital or data strategy may count only if open data is clearly and substantively included.",
        "P3": "Look for regional, local, municipal, or city-level open data policy or strategy evidence. National-only evidence is not enough on its own.",
        "P4": "Look for an action plan, roadmap, implementation measures, milestones, or operational delivery plan linked to the open data strategy or policy.",
        "P13": "Focus on governance structures, responsible ministries or agencies, steering groups, coordination mechanisms, and stakeholder governance models.",
        "P16": "This is an extent question, not a simple yes/no question. Distinguish carefully between none, few, and many. Use breadth of local/regional evidence across public bodies, municipalities, or regions. Do not choose a stronger category unless the evidence clearly supports that breadth.",
        "P22": "Focus on data publication plans, dataset release planning, publication obligations, or official guidance for public bodies.",
        "P24": "Look for review processes, updates, revised strategies, or formal procedures for updating open data policy or strategy.",
        "I1": "Look for an explicit or near-explicit definition of open data reuse in methodology, policy, or monitoring documents. Do not treat generic examples of reuse as a formal definition unless the evidence clearly defines the concept.",
        "I2": "Look for processes, methodologies, dashboards, reports, or official mechanisms used to monitor reuse.",
        "I9": "Look for consultations, workshops, surveys, engagement events, or user-needs assessments related to stimulating reuse.",
        "I10": "Look for systematic collection of reuse cases, official showcases, or recurring reporting on reuse examples. One isolated example is weaker than a systematic collection process.",
        "I12": "Look for documented governmental impact, such as efficiency, transparency, decision-making improvement, or better public services. Prefer studies, reports, or concrete official examples over broad claims.",
        "I17": "Look for social impact evidence, including inclusion, health, education, awareness, or public-value outcomes. Prefer documented examples or reports over general statements.",
        "I27": "Look for economic impact evidence such as innovation, business reuse, startups, market value, productivity, or economic reports. Prefer documented evidence, studies, or official examples over broad claims that open data is economically useful.",
    }

    if qid in hints:
        return hints[qid]

    if dim == "policy":
        return "Focus on laws, strategies, decrees, governance structures, official roles, and implementation processes."
    return "Focus on definitions, monitoring, reuse activities, impact reports, methodologies, and evidence of outcomes."


def is_belgium_compact_question(country: str, question_id: str) -> bool:
    return safe_text(country).lower() == "belgium" and safe_text(question_id) in {"I10", "P24", "P2", "I2", "I1"}


def build_style_instruction(country: str, dimension: str, question_id: str) -> str:
    dim = safe_text(dimension).lower()
    qid = safe_text(question_id)

    # Belgium-specific compact style for short pointer-like benchmark questions
    if is_belgium_compact_question(country, question_id):
        return """
Write the justification in a very compact benchmark-like style.

Required writing style:
- Keep the explanation short, usually 45 to 100 words.
- Use 1 short paragraph or at most 2 very short paragraphs.
- Start directly with the answer and strongest evidence.
- Prefer concise document-oriented wording over narrative explanation.
- Mention only the most relevant law, strategy, report, or official source.
- Avoid broad contextual discussion.
- Avoid repeating the question.
- Do NOT cite sources in the body using S1, [S1], or similar markers.
- Reserve explicit listing for the final ## Sources section only.
- End with a section exactly titled: ## Sources
- Under ## Sources, list the sources in this format:
(1): Title
URL
""".strip()

    base = """
Write the justification in a compact benchmark-like style.

Required writing style:
- Start with a direct answer in the first sentence, such as "Yes, ...", "Partially, ...", "No, ...", or the equivalent of the selected response.
- Keep the justification concise, usually around 90 to 170 words unless the evidence is genuinely complex.
- Prefer 2 to 4 short evidence-focused paragraphs or numbered points.
- Prioritise only the strongest evidence; do not include broad background discussion.
- Write naturally and clearly, not in a robotic or overly defensive tone.
- Do not begin with phrases like "No evidence..." unless the retrieved evidence is genuinely too weak to support any real answer.
- Do not mention "Open Data Maturity", "ODM", "benchmark", or "assessment framework" in the justification.
- Write as if you are independently summarising country evidence, not describing the benchmark.
- Mention the names of laws, decrees, ministries, strategies, reports, methodologies, programmes, or official initiatives where supported.
- Focus on concrete evidence statements rather than generic commentary.
- Avoid filler, avoid repeating the question, and avoid long introductory framing.
- Do NOT cite sources in the body using identifiers such as S1, S2, [S1], (S1), or similar source-number markers.
- If you mention evidence in the body, mention the document or initiative name naturally instead of using source IDs.
- Reserve all explicit source listing for the final ## Sources section only.
- End with a section exactly titled: ## Sources
- Under ## Sources, list the sources in this format:
(1): Title
URL
"""

    if dim == "policy":
        extra = """
For Policy questions, a strong compact structure is:
1. Direct answer
2. Legal or strategic basis
3. Governance or implementation support
"""
    else:
        extra = """
For Impact questions, a strong compact structure is:
1. Direct answer
2. Monitoring / methodology / reuse evidence
3. Concrete government, social, or economic impact evidence
"""

    return (base + "\n" + extra).strip()


def build_prompt(
    country: str,
    question_id: str,
    dimension: str,
    question: str,
    response_options: list[str],
    score_map: dict,
    evidence_block: str,
) -> str:
    style_instruction = build_style_instruction(country, dimension, question_id)
    question_hint = get_question_hint(question_id, dimension)

    response_options_text = "\n".join([f"- {opt}" for opt in response_options]) if response_options else "- No official response options available"

    return f"""
You are assisting with a country-level open data policy and impact evidence review.

Country: {country}
Question ID: {question_id}
Dimension: {dimension}
Question: {question}

Official response options:
{response_options_text}

Official response scoring map:
{format_score_map(score_map)}

Question-specific assessment hint:
{question_hint}

Use ONLY the evidence below.
Do not invent facts.
Do not use outside knowledge.

Important instructions:
- Choose the best official response option from the official response options when they are available.
- Select the strongest official answer that is clearly supported by the evidence.
- Do not exaggerate beyond the evidence, but also do not understate when the evidence is strong enough.
- If the evidence is mixed or incomplete, choose carefully and explain the limitation clearly.
- Your score_suggestion must align with the official response scoring map.
- The justification should sound like a compact human-written evidence synthesis, not a refusal or benchmark description.

{style_instruction}

Return JSON only in this format:
{{
  "final_answer_selected": "...",
  "score_suggestion": 0,
  "justification_generated": "...",
  "confidence": 0,
  "evidence_sufficiency": "sufficient / partial / weak"
}}

Evidence:
{evidence_block}
""".strip()


def build_second_stage_selector_prompt(
    country: str,
    question_id: str,
    dimension: str,
    question: str,
    response_options: list[str],
    score_map: dict,
    first_stage_answer: str,
    first_stage_justification: str,
    evidence_block: str,
) -> str:
    response_options_text = "\n".join([f"- {opt}" for opt in response_options]) if response_options else "- No official response options available"

    return f"""
You are performing a final constrained answer selection for an ODM-style benchmark question.

Country: {country}
Question ID: {question_id}
Dimension: {dimension}
Question: {question}

Official response options:
{response_options_text}

Official response scoring map:
{format_score_map(score_map)}

First-stage proposed answer:
{first_stage_answer}

First-stage justification:
{first_stage_justification}

Evidence:
{evidence_block}

Task:
Choose exactly one final official response option from the official response options above.

Rules:
- Output only one option from the official response options if such options are available.
- Choose the strongest option clearly supported by the evidence.
- Do not be weaker than the evidence justifies.
- Do not be stronger than the evidence justifies.
- Prefer exact official option wording.
- If the evidence supports only a limited or partial conclusion, choose the corresponding partial option if available.

Return JSON only in this format:
{{
  "final_answer_selected": "..."
}}
""".strip()


def build_sources_section_from_retrieved_df(qdf: pd.DataFrame) -> str:
    used = qdf.head(MAX_EVIDENCE_CHUNKS).copy() if not qdf.empty else pd.DataFrame()

    lines = ["## Sources"]

    if used.empty:
        lines.append("No retrieved sources available.")
        return "\n".join(lines)

    seen = set()
    source_num = 1

    for _, row in used.iterrows():
        title = safe_text(row.get("source_title", "")) or "Untitled source"
        url = safe_text(row.get("source_url", "")) or "URL not available"

        dedup_key = (title.lower(), url.lower())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        lines.append(f"({source_num}): {title}")
        lines.append(url)
        source_num += 1

    if source_num == 1:
        lines.append("No retrieved sources available.")

    return "\n".join(lines)


def strip_existing_sources_section(justification: str) -> str:
    text = safe_text(justification)
    if not text:
        return ""

    parts = re.split(r"(?i)##\s*Sources", text, maxsplit=1)
    return safe_text(parts[0])


def remove_inline_source_markers(text: str) -> str:
    body = safe_text(text)
    if not body:
        return ""

    patterns = [
        r"\[\s*S\d+(?:\s*,\s*S\d+)*\s*\]",
        r"\(\s*S\d+(?:\s*,\s*S\d+)*\s*\)",
        r"\bS\d+(?:\s*,\s*S\d+)*\b",
    ]

    for pattern in patterns:
        body = re.sub(pattern, "", body, flags=re.IGNORECASE)

    body = re.sub(r"\s{2,}", " ", body)
    body = re.sub(r"\(\s*\)", "", body)
    body = re.sub(r"\s+([,.;:])", r"\1", body)
    body = re.sub(r"\n[ \t]+\n", "\n\n", body)

    return body.strip()


def enforce_sources_section(justification: str, qdf: pd.DataFrame) -> str:
    body = strip_existing_sources_section(justification)
    body = remove_inline_source_markers(body)
    sources_section = build_sources_section_from_retrieved_df(qdf)

    if body:
        return f"{body}\n\n{sources_section}"
    return sources_section


def run_second_stage_selector(
    client: OpenAI,
    country: str,
    question_id: str,
    dimension: str,
    question_text: str,
    response_options: list[str],
    score_map: dict,
    first_stage_answer: str,
    first_stage_justification: str,
    qdf: pd.DataFrame,
) -> str:
    if not response_options:
        return safe_text(first_stage_answer)

    evidence_block = build_evidence_block(qdf)

    prompt = build_second_stage_selector_prompt(
        country=country,
        question_id=question_id,
        dimension=dimension,
        question=question_text,
        response_options=response_options,
        score_map=score_map,
        first_stage_answer=first_stage_answer,
        first_stage_justification=first_stage_justification,
        evidence_block=evidence_block,
    )

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
        )
        parsed = parse_json_output(response.output_text)
        selected = safe_text(parsed.get("final_answer_selected", ""))
        return validate_selected_response(selected, response_options)
    except Exception as e:
        print(f"Second-stage selector failed for {country} | {question_id}: {e}")
        return validate_selected_response(first_stage_answer, response_options)


def fallback_output(question: str) -> dict:
    return {
        "final_answer_selected": "",
        "score_suggestion": None,
        "justification_generated": (
            f"The retrieved evidence was too limited to support a reliable answer for this question: {question}"
        ),
        "confidence": 10,
        "evidence_sufficiency": "weak",
    }


def generate_answer_for_question(
    country: str,
    question_id: str,
    question_text: str,
    dimension: str,
    response_scoring: str,
    retrieved_df: pd.DataFrame,
    client: OpenAI = None,
) -> dict:
    country = safe_text(country)
    question_id = safe_text(question_id)
    question_text = safe_text(question_text)
    dimension = safe_text(dimension)
    response_scoring = safe_text(response_scoring)

    if client is None:
        client = get_openai_client()

    score_map = parse_response_scoring(response_scoring)
    response_options = response_option_list(score_map)

    qdf = retrieved_df.copy().sort_values(by="retrieval_rank") if not retrieved_df.empty else pd.DataFrame()

    if qdf.empty:
        result = fallback_output(question_text)
    else:
        evidence_block = build_evidence_block(qdf)

        prompt = build_prompt(
            country=country,
            question_id=question_id,
            dimension=dimension,
            question=question_text,
            response_options=response_options,
            score_map=score_map,
            evidence_block=evidence_block,
        )

        try:
            response = client.responses.create(
                model=MODEL_NAME,
                input=prompt,
            )
            result = parse_json_output(response.output_text)
            if not result:
                result = fallback_output(question_text)
        except Exception as e:
            print(f"Generation failed for {country} | {question_id}: {e}")
            result = fallback_output(question_text)

    first_stage_answer = validate_selected_response(
        result.get("final_answer_selected", ""),
        response_options
    )

    raw_justification = safe_text(result.get("justification_generated", ""))
    clean_justification_body = strip_existing_sources_section(raw_justification)
    clean_justification_body = remove_inline_source_markers(clean_justification_body)

    final_answer_selected = run_second_stage_selector(
        client=client,
        country=country,
        question_id=question_id,
        dimension=dimension,
        question_text=question_text,
        response_options=response_options,
        score_map=score_map,
        first_stage_answer=first_stage_answer,
        first_stage_justification=clean_justification_body,
        qdf=qdf,
    )

    final_answer_selected = canonicalize_final_answer(
        final_answer_selected,
        response_scoring=score_map,
    )

    score_suggestion = result.get("score_suggestion", None)
    official_score = infer_score_from_selected_response(final_answer_selected, score_map)

    if official_score is not None:
        score_suggestion = official_score

    final_justification = enforce_sources_section(clean_justification_body, qdf)

    return {
        "final_answer_selected": final_answer_selected,
        "score_suggestion": score_suggestion,
        "justification_generated": final_justification,
        "confidence": result.get("confidence", None),
        "evidence_sufficiency": safe_text(result.get("evidence_sufficiency", "")),
        "retrieved_evidence_count": len(qdf),
        "response_scoring_dict": score_map,
    }


def main():
    print_header("STEP 10 EXPERIMENT: GENERATE ODM ANSWERS FROM CHROMA RETRIEVAL")

    ground_truth_path = PROCESSED_DIR / "ground_truth_odm_selected.csv"
    registry_path = PROCESSED_DIR / "registry_odm_selected_questions.csv"
    retrieval_path = INTERIM_DIR / "retrieval" / "retrieved_topk_odm_selected_chroma.csv"

    output_dir = GENERATION_DIR
    output_path = output_dir / "generated_answers_odm_selected_chroma.csv"

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {ground_truth_path}\n"
            f"Please run step2_build_ground_truth.py first."
        )

    if not registry_path.exists():
        raise FileNotFoundError(
            f"Registry file not found: {registry_path}\n"
            f"Please run step1_build_odm_policy_registry.py first."
        )

    if not retrieval_path.exists():
        raise FileNotFoundError(
            f"Chroma retrieval file not found: {retrieval_path}\n"
            f"Please run step9_retrieve_topk_chroma.py first."
        )

    ensure_dir(output_dir)

    gt_df = pd.read_csv(ground_truth_path)
    registry_df = pd.read_csv(registry_path)
    retrieval_df = pd.read_csv(retrieval_path)

    registry_df = registry_df.drop_duplicates(subset=["question_id"]).copy()

    target_df = gt_df[["country", "question_id", "dimension", "question"]].drop_duplicates().copy()
    target_df = target_df.merge(
        registry_df[["question_id"] + ([col for col in ["response_scoring"] if col in registry_df.columns])],
        on="question_id",
        how="left"
    )

    client = get_openai_client()
    rows = []

    total = len(target_df)

    for idx, (_, row) in enumerate(target_df.iterrows(), start=1):
        country = safe_text(row.get("country", ""))
        question_id = safe_text(row.get("question_id", ""))
        dimension = safe_text(row.get("dimension", ""))
        question_text = safe_text(row.get("question", ""))
        response_scoring = safe_text(row.get("response_scoring", ""))

        qdf = retrieval_df[
            (retrieval_df["country"].astype(str).str.strip() == country) &
            (retrieval_df["question_id"].astype(str).str.strip() == question_id)
        ].copy()

        print(f"[{idx}/{total}] Generating from Chroma retrieval for {country} | {question_id} | evidence_rows={len(qdf)}")

        result = generate_answer_for_question(
            country=country,
            question_id=question_id,
            question_text=question_text,
            dimension=dimension,
            response_scoring=response_scoring,
            retrieved_df=qdf,
            client=client,
        )

        rows.append({
            "country": country,
            "question_id": question_id,
            "dimension": dimension,
            "question": question_text,
            "response_scoring": response_scoring,
            "final_answer_selected": result.get("final_answer_selected", ""),
            "score_suggestion": result.get("score_suggestion", None),
            "justification_generated": result.get("justification_generated", ""),
            "confidence": result.get("confidence", None),
            "evidence_sufficiency": result.get("evidence_sufficiency", ""),
            "mode": "chroma_retrieval_experiment",
        })

    out_df = pd.DataFrame(rows)

    preferred_cols = [
        "country",
        "question_id",
        "dimension",
        "question",
        "response_scoring",
        "final_answer_selected",
        "score_suggestion",
        "justification_generated",
        "confidence",
        "evidence_sufficiency",
        "mode",
    ]
    existing_cols = [c for c in preferred_cols if c in out_df.columns]
    remaining_cols = [c for c in out_df.columns if c not in existing_cols]
    out_df = out_df[existing_cols + remaining_cols]

    save_csv(out_df, output_path)

    print(f"\nSaved Chroma generated answers to: {output_path}")
    print(f"Rows: {len(out_df)}")

    if len(out_df) > 0:
        print("\nPreview:")
        preview_cols = [
            c for c in [
                "country", "question_id", "final_answer_selected",
                "score_suggestion", "confidence", "evidence_sufficiency", "mode"
            ] if c in out_df.columns
        ]
        print(out_df[preview_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()