import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st

from country_flags import get_flag_url
from export_utils import build_result_pdf_bytes


def safe_text(value) -> str:
    return str(value or "").strip()


def normalize_dimension_label(value: str) -> str:
    raw = safe_text(value).lower()
    if raw in {"policy_dimension", "policy"}:
        return "Policy"
    if raw in {"impact_dimension", "impact"}:
        return "Impact"
    return safe_text(value)


def friendly_answer_text(result: dict) -> str:
    answer = str(result.get("final_answer_selected", "") or "").strip()

    if answer.lower() == "no official options available":
        return "Open-ended synthesis"

    if not answer:
        return "No answer"

    return answer


def friendly_mode_text(mode: str) -> str:
    mapping = {
        "live_main_pipeline": "Live retrieval",
        "unsupported": "Unsupported request",
    }
    return mapping.get(mode, mode.replace("_", " ").title())


def format_confidence_100(value) -> str:
    if value is None or value == "":
        return "—"

    try:
        v = float(value)
    except Exception:
        return str(value)

    if v <= 1:
        v = v * 100

    v = max(0, min(v, 100))
    return f"{v:.0f}/100"


def format_score_display(result: dict) -> str:
    score_val = result.get("score_suggestion", None)
    response_scoring = result.get("response_scoring", None)

    if score_val is None or score_val == "":
        return "—"

    try:
        score_num = float(score_val)
    except Exception:
        return str(score_val)

    max_score = None

    if isinstance(response_scoring, dict) and response_scoring:
        numeric_values = []
        for v in response_scoring.values():
            try:
                numeric_values.append(float(v))
            except Exception:
                continue
        if numeric_values:
            max_score = max(numeric_values)

    if max_score is not None:
        if score_num.is_integer():
            score_text = str(int(score_num))
        else:
            score_text = f"{score_num:.1f}"

        if float(max_score).is_integer():
            max_text = str(int(max_score))
        else:
            max_text = f"{max_score:.1f}"

        return f"{score_text} / {max_text}"

    if score_num.is_integer():
        return str(int(score_num))

    return f"{score_num:.1f}"


def apply_custom_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f5f7fc;
            color: #17305b;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1250px;
        }

        h1, h2, h3 {
            color: #123b8f;
        }

        .odm-banner {
            background: linear-gradient(135deg, #0f2f7f 0%, #1b4fb8 100%);
            color: white;
            padding: 1.2rem 1.4rem;
            border-radius: 18px;
            margin-bottom: 1rem;
            box-shadow: 0 8px 20px rgba(18, 59, 143, 0.15);
        }

        .odm-banner h1 {
            color: white !important;
            margin: 0 0 0.5rem 0;
            font-size: 2rem;
        }

        .odm-banner p {
            margin: 0;
            font-size: 1rem;
            opacity: 0.95;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 12px;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        }

        .summary-card {
            background: white;
            border-left: 5px solid #f6c431;
            border-radius: 14px;
            padding: 12px 14px;
            box-shadow: 0 6px 14px rgba(18, 59, 143, 0.08);
        }

        .summary-label {
            font-size: 0.78rem;
            color: #58739b;
            margin-bottom: 0.2rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }

        .summary-value {
            font-size: 1.1rem;
            color: #17305b;
            font-weight: 700;
            line-height: 1.3;
        }

        .question-card {
            background: white;
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 8px 18px rgba(18, 59, 143, 0.08);
            margin-bottom: 1rem;
        }

        .question-country-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 0.6rem;
        }

        .question-country-row img {
            border-radius: 4px;
            width: 38px;
            height: auto;
        }

        .question-country-name {
            font-size: 1.75rem;
            font-weight: 800;
            color: #123b8f;
        }

        .question-meta {
            font-size: 1rem;
            color: #36527d;
            margin-bottom: 0.8rem;
        }

        .question-text-big {
            font-size: 1.08rem;
            line-height: 1.7;
            color: #17305b;
            font-weight: 500;
        }

        .section-card {
            background: transparent;
            border-radius: 0;
            padding: 0;
            box-shadow: none;
            margin-bottom: 1rem;
        }

        .small-note {
            color: #627ca2;
            font-size: 0.92rem;
            margin-top: 0.2rem;
        }

        .center-loader-wrap {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 24px 0 8px 0;
        }

        .center-loader {
            width: 44px;
            height: 44px;
            border: 5px solid #d9e4fb;
            border-top: 5px solid #1b4fb8;
            border-radius: 50%;
            animation: odmspin 0.9s linear infinite;
            margin-bottom: 10px;
        }

        @keyframes odmspin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .history-caption {
            color: #58739b;
            font-size: 0.92rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_center_loader(message: str = "Processing..."):
    st.markdown(
        f"""
        <div class="center-loader-wrap">
            <div class="center-loader"></div>
            <div style="color:#36527d;font-weight:600;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_app_banner():
    st.markdown(
        """
        <div class="odm-banner">
            <h1>ODM Policy &amp; Impact Explorer</h1>
            <p>
                Explore Open Data Maturity policy and impact questions by country.
                Every run performs live evidence retrieval from the current web.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_header(result: dict):
    answer = friendly_answer_text(result)
    score_val = format_score_display(result)
    confidence_val = format_confidence_100(result.get("confidence", None))
    source_count = len(result.get("retrieved_sources", []) or [])

    html = f"""
    <div class="summary-grid" style="grid-template-columns: repeat(4, minmax(0, 1fr));">
        <div class="summary-card">
            <div class="summary-label">Answer</div>
            <div class="summary-value">{answer}</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Question Score</div>
            <div class="summary-value">{score_val}</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">LLM Confidence</div>
            <div class="summary-value">{confidence_val}</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Total Sources</div>
            <div class="summary-value">{source_count}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    st.markdown(
        f"<div class='small-note'>Evidence sufficiency: {result.get('evidence_sufficiency', '')}</div>",
        unsafe_allow_html=True,
    )


def render_question_info(result: dict):
    country = result.get("country", "")
    question_id = result.get("question_id", "")
    dimension = normalize_dimension_label(result.get("dimension", ""))
    question = result.get("question", "")
    flag_url = get_flag_url(country)

    flag_html = f'<img src="{flag_url}" alt="flag" />' if flag_url else ""

    html = f"""
    <div class="question-card">
        <div class="question-country-row">
            {flag_html}
            <div class="question-country-name">{country}</div>
        </div>
        <div class="question-meta">
            <strong>Question ID:</strong> {question_id} &nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Dimension:</strong> {dimension}
        </div>
        <div class="question-text-big">{question}</div>
    </div>
    """
    st.markdown("### Question")
    st.markdown(html, unsafe_allow_html=True)


def split_justification_and_sources(justification: str) -> tuple[str, list[str]]:
    text = str(justification or "").strip()
    if not text:
        return "", []

    match = re.split(r"(?i)##\s*Sources", text, maxsplit=1)
    body = match[0].strip()

    if len(match) == 1:
        return body, []

    raw_sources = match[1].strip()
    source_lines = []

    for line in raw_sources.splitlines():
        clean = line.strip()
        if clean:
            source_lines.append(clean)

    return body, source_lines


def render_justification(result: dict):
    st.markdown("### Justification")
    justification = result.get("justification_generated", "")
    body, source_lines = split_justification_and_sources(justification)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if body:
        st.write(body)
    else:
        st.info("No justification available.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Sources")
    if source_lines:
        for line in source_lines:
            st.write(line)
    else:
        st.info("No model-written sources section available.")


def clean_snippet_text(text: str) -> str:
    snippet = str(text or "").strip()
    if not snippet:
        return ""

    lines = []
    for line in snippet.splitlines():
        cleaned = line.strip()
        cleaned = re.sub(r"^\\-\s*", "", cleaned)
        cleaned = re.sub(r"^-\s*", "", cleaned)
        if cleaned:
            lines.append(cleaned)

    return "\n".join(lines).strip()


def render_sources(result: dict):
    sources = result.get("retrieved_sources", [])
    st.markdown("### Retrieved Sources")

    if not sources:
        st.info("No retrieved sources available.")
        return

    for source in sources:
        rank = source.get("retrieval_rank", "")
        title = source.get("source_title", "") or "Untitled source"
        url = source.get("source_url", "")
        file_type = source.get("source_file_type", "")
        snippet = clean_snippet_text(source.get("chunk_text", ""))

        with st.expander(f"[{rank}] {title}"):
            st.write(f"**File type:** {file_type}")

            if url:
                st.markdown(f"**URL:** [{url}]({url})")
            else:
                st.write("**URL:** N/A")

            if snippet:
                st.write("**Snippet:**")
                st.text(snippet[:2000])


def format_created_at(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""

    app_tz = os.getenv("APP_TIMEZONE", "Europe/London")

    try:
        dt_utc = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        dt_local = dt_utc.astimezone(ZoneInfo(app_tz))
        return dt_local.strftime("%d/%m/%Y %H:%M")
    except Exception:
        return raw


def render_saved_runs_rows(rows: list[dict], container_height: int = 420) -> bool:
    if not rows:
        st.info("No saved runs found.")
        return False

    st.markdown("### Saved run list")
    st.markdown(
        f"<div class='history-caption'>Showing all {len(rows)} saved runs in a scrollable window.</div>",
        unsafe_allow_html=True,
    )

    any_deleted = False

    with st.container(height=container_height, border=True):
        header_cols = st.columns([0.6, 1.8, 1.3, 1.0, 1.1, 0.9, 0.9])
        headers = ["No.", "Saved Time", "Country", "Question ID", "Dimension", "Open", "Delete"]
        for col, text in zip(header_cols, headers):
            col.markdown(f"**{text}**")

        st.markdown("---")

        for display_idx, row in enumerate(rows, start=1):
            cols = st.columns([0.6, 1.8, 1.3, 1.0, 1.1, 0.9, 0.9])

            display_dimension = normalize_dimension_label(row.get("dimension", ""))

            cols[0].write(display_idx)
            cols[1].write(format_created_at(row.get("created_at", "")))
            cols[2].write(row.get("country", ""))
            cols[3].write(row.get("question_id", ""))
            cols[4].write(display_dimension)

            real_id = row.get("id")

            if cols[5].button("Open", key=f"open_{real_id}"):
                st.session_state["load_saved_id"] = real_id

            if cols[6].button("Delete", key=f"delete_{real_id}"):
                st.session_state["delete_saved_id"] = real_id
                any_deleted = True

    return any_deleted


def render_download_buttons(result: dict, key_suffix: str = "default"):
    st.markdown("### Export")
    pdf_bytes = build_result_pdf_bytes(result)

    country = str(result.get("country", "result") or "result").strip()
    question_id = str(result.get("question_id", "question") or "question").strip()

    st.download_button(
        label="Download result as PDF",
        data=pdf_bytes,
        file_name=f"{country}_{question_id}.pdf",
        mime="application/pdf",
        key=f"download_pdf_{country}_{question_id}_{key_suffix}",
    )


def render_full_result(result: dict, key_suffix: str = "default"):
    render_result_header(result)
    render_question_info(result)
    render_justification(result)
    render_sources(result)
    render_download_buttons(result, key_suffix=key_suffix)