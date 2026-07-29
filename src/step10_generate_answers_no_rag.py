import ast
import json
import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from config import PROCESSED_DIR, GENERATION_DIR
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


def validate_selected_response(selected_response: str, response_options: list[str]) -> str:
    selected_response = safe_text(selected_response)

    if not response_options:
        return selected_response

    option_map = {opt.lower(): opt for opt in response_options}

    if selected_response.lower() in option_map:
        return option_map[selected_response.lower()]

    for opt in response_options:
        if selected_response.lower() == safe_text(opt).lower():
            return opt

    return selected_response


def infer_score_from_selected_response(selected_response: str, score_map: dict):
    if not score_map:
        return None

    selected_response = safe_text(selected_response)
    for opt, score in score_map.items():
        if selected_response.lower() == safe_text(opt).lower():
            return score

    return None


def get_question_hint(question_id: str, dimension: str) -> str:
    qid = safe_text(question_id)
    dim = safe_text(dimension).lower()

    hints = {
        "P1": "Focus on whether there is a formal national open data policy framework and legal basis, including directive transposition where relevant.",
        "P2": "Focus on whether there is an explicit national open data strategy or roadmap, not just broad digital policy.",
        "P3": "Focus on regional or local open data policy evidence, not only national-level evidence.",
        "P4": "Focus on action plans, implementation measures, milestones, or roadmaps linked to open data policy.",
        "P13": "Focus on governance model, responsible institutions, steering groups, and coordination mechanisms.",
        "P16": "This is an extent question. Distinguish carefully between none, few, and many local/regional public bodies.",
        "P22": "Focus on publication plans, data release planning, and formal publication obligations.",
        "P24": "Focus on review, update, or revision processes for policy or strategy.",
        "I1": "Focus on whether reuse is explicitly or near-explicitly defined.",
        "I2": "Focus on whether there are processes or methodologies for monitoring reuse.",
        "I9": "Focus on activities to understand reusers’ needs, such as consultations, workshops, or surveys.",
        "I10": "Focus on whether reuse cases are collected systematically, not just isolated examples.",
        "I12": "Focus on documented governmental impact, such as efficiency, transparency, or decision-making improvement.",
        "I17": "Focus on documented social impact, not generic benefit claims.",
        "I27": "Focus on documented economic impact, such as innovation, business value, or economic studies.",
    }

    if qid in hints:
        return hints[qid]

    if dim == "policy":
        return "Focus on laws, strategies, governance, and implementation."
    return "Focus on reuse, monitoring, impact, and documented outcomes."


def build_prompt(country: str, question_id: str, dimension: str, question: str, response_options: list[str], score_map: dict) -> str:
    response_options_text = "\n".join([f"- {opt}" for opt in response_options]) if response_options else "- No official options available"
    question_hint = get_question_hint(question_id, dimension)

    return f"""
You are answering a country-level open data policy and impact assessment question WITHOUT retrieved evidence.

Country: {country}
Question ID: {question_id}
Dimension: {dimension}
Question: {question}

Official response options:
{response_options_text}

Official response scoring map:
{format_score_map(score_map)}

Question-specific hint:
{question_hint}

Important instructions:
- This is a no-retrieval baseline. Do not pretend you have retrieved evidence.
- Answer using only your general prior knowledge and reasoning.
- If you are uncertain, answer conservatively.
- Choose exactly one official response option when options are available.
- Your score_suggestion must align with the official response scoring map.
- Write a short, natural justification.
- Do not mention Open Data Maturity, ODM, benchmark, or assessment framework.
- Do not fabricate URLs or sources.
- End with:
## Sources
No retrieved sources (no-RAG baseline).

Return JSON only in this format:
{{
  "final_answer_selected": "...",
  "score_suggestion": 0,
  "justification_generated": "...",
  "confidence": 0,
  "evidence_sufficiency": "sufficient / partial / weak"
}}
""".strip()


def fallback_output(question: str) -> dict:
    return {
        "final_answer_selected": "",
        "score_suggestion": None,
        "justification_generated": (
            f"A reliable no-RAG answer could not be generated for this question: {question}\n\n"
            f"## Sources\n"
            f"No retrieved sources (no-RAG baseline)."
        ),
        "confidence": 5,
        "evidence_sufficiency": "weak",
    }


def main():
    print_header("STEP 10 (NO-RAG): GENERATE ODM ANSWERS")

    ground_truth_path = PROCESSED_DIR / "ground_truth_odm_selected.csv"
    registry_path = PROCESSED_DIR / "registry_odm_selected_questions.csv"

    output_dir = GENERATION_DIR
    output_path = output_dir / "generated_answers_odm_selected_no_rag.csv"

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

    ensure_dir(output_dir)

    gt_df = pd.read_csv(ground_truth_path)
    registry_df = pd.read_csv(registry_path).drop_duplicates(subset=["question_id"]).copy()

    target_df = gt_df[["country", "question_id", "dimension", "question"]].drop_duplicates().copy()
    target_df = target_df.merge(
        registry_df[["question_id"] + ([col for col in ["response_scoring"] if col in registry_df.columns])],
        on="question_id",
        how="left"
    )

    client = get_openai_client()
    rows = []

    for _, target in target_df.iterrows():
        country = safe_text(target.get("country", ""))
        question_id = safe_text(target.get("question_id", ""))
        dimension = safe_text(target.get("dimension", ""))
        question = safe_text(target.get("question", ""))
        response_scoring_raw = safe_text(target.get("response_scoring", ""))

        score_map = parse_response_scoring(response_scoring_raw)
        response_options = response_option_list(score_map)

        prompt = build_prompt(
            country=country,
            question_id=question_id,
            dimension=dimension,
            question=question,
            response_options=response_options,
            score_map=score_map,
        )

        try:
            response = client.responses.create(
                model=MODEL_NAME,
                input=prompt,
            )
            result = parse_json_output(response.output_text)
            if not result:
                result = fallback_output(question)
        except Exception as e:
            print(f"No-RAG generation failed for {country} | {question_id}: {e}")
            result = fallback_output(question)

        final_answer_selected = validate_selected_response(
            result.get("final_answer_selected", ""),
            response_options
        )

        score_suggestion = result.get("score_suggestion", None)
        official_score = infer_score_from_selected_response(final_answer_selected, score_map)
        if official_score is not None:
            score_suggestion = official_score

        rows.append({
            "country": country,
            "question_id": question_id,
            "dimension": dimension,
            "question": question,
            "response_scoring": response_scoring_raw,
            "final_answer_selected": final_answer_selected,
            "score_suggestion": score_suggestion,
            "justification_generated": safe_text(result.get("justification_generated", "")),
            "confidence": result.get("confidence", None),
            "evidence_sufficiency": safe_text(result.get("evidence_sufficiency", "")),
            "retrieved_evidence_count": 0,
        })

        print(
            f"No-RAG generated answer for {country} | {question_id} | "
            f"selected={final_answer_selected} | score={score_suggestion}"
        )

    out_df = pd.DataFrame(rows)
    save_csv(out_df, output_path)

    print(f"\nSaved no-RAG generated answers to: {output_path}")
    print(f"Rows: {len(out_df)}")

    if len(out_df) > 0:
        print("\nPreview:")
        print(
            out_df[[
                "country", "question_id", "final_answer_selected",
                "score_suggestion", "confidence", "evidence_sufficiency"
            ]].head(20).to_string(index=False)
        )


if __name__ == "__main__":
    main()