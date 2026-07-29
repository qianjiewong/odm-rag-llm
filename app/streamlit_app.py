import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# On Streamlit Community Cloud, credentials are provided via st.secrets
# (configured in the app dashboard), not a local .env file. The rest of
# this codebase reads credentials with os.getenv(...), so bridge any
# secrets Streamlit has loaded into the process environment here, once,
# before any pipeline module is imported. Locally, st.secrets is simply
# empty and this loop does nothing — .env file continues to work
# exactly as before via python-dotenv.
for _key, _value in st.secrets.items():
    os.environ.setdefault(_key, str(_value))

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
APP_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from pipeline_runner import MainQuestionPipeline
from db import (
    init_db,
    save_search_result,
    list_saved_results_filtered,
    get_saved_result,
    delete_saved_result,
)
from ui_helpers import (
    apply_custom_theme,
    render_app_banner,
    render_full_result,
    render_saved_runs_rows,
)

st.set_page_config(
    page_title="ODM Policy & Impact Explorer",
    page_icon="🔵",
    layout="wide",
)

init_db()
apply_custom_theme()
render_app_banner()

with st.expander("How it works"):
    st.markdown(
        """
1. Select a country and one ODM policy or impact question.  
2. The app calls the same main question-by-question pipeline used by the evaluation system.  
3. The pipeline retrieves top-k evidence from the prepared evidence base.  
4. The LLM generates a grounded justification from those retrieved chunks.  
5. The result is shown in the UI and saved to history for later review.
"""
    )

PROCESSED_DIR = (BASE_DIR / "data" / "processed").resolve()
selected_ground_truth_path = PROCESSED_DIR / "ground_truth_odm_selected.csv"
demo_registry_path = PROCESSED_DIR / "registry_odm_demo_policy_impact.csv"
selected_registry_path = PROCESSED_DIR / "registry_odm_selected_questions.csv"

if not selected_ground_truth_path.exists():
    st.error(f"Ground truth file not found: {selected_ground_truth_path}")
    st.stop()

if demo_registry_path.exists():
    registry_df = pd.read_csv(demo_registry_path)
else:
    registry_df = pd.read_csv(selected_registry_path)

ground_truth_df = pd.read_csv(selected_ground_truth_path)

ALL_ODM_COUNTRIES = sorted([
    "Albania", "Austria", "Belgium", "Bosnia and Herzegovina", "Bulgaria",
    "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland", "France",
    "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Latvia",
    "Lithuania", "Luxembourg", "Malta", "Montenegro", "Netherlands",
    "North Macedonia", "Norway", "Poland", "Portugal", "Romania", "Serbia",
    "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Ukraine",
])

evaluated_countries = sorted(ground_truth_df["country"].dropna().astype(str).unique().tolist())
demo_country_options = ALL_ODM_COUNTRIES


def is_extended_question_id(question_id: str) -> bool:
    qid = str(question_id or "").strip().lower()
    return bool(re.search(r"\d+[a-z]$|[-_/]", qid))


def numeric_question_sort_key(question_id: str):
    qid = str(question_id or "").strip()
    match = re.search(r"(\d+)", qid)
    num = int(match.group(1)) if match else 9999
    prefix = re.sub(r"\d+", "", qid).strip().lower()
    return (num, prefix, qid.lower())


def dimension_sort_key(dimension: str) -> int:
    dim = str(dimension or "").strip().lower()
    if dim == "policy":
        return 0
    if dim == "impact":
        return 1
    return 9


question_options_df = registry_df[["question_id", "dimension", "question"]].drop_duplicates().copy()
question_options_df = question_options_df[
    ~question_options_df["question_id"].astype(str).apply(is_extended_question_id)
].copy()

question_options_df["dimension"] = question_options_df["dimension"].astype(str).replace({
    "policy_dimension": "Policy",
    "impact_dimension": "Impact",
    "policy": "Policy",
    "impact": "Impact",
})
question_options_df["dimension_order"] = question_options_df["dimension"].map(dimension_sort_key)

question_options_df = question_options_df.sort_values(
    by=["dimension_order", "question_id"],
    key=lambda col: col.map(numeric_question_sort_key) if col.name == "question_id" else col
).reset_index(drop=True)

question_options_df["display_order"] = range(1, len(question_options_df) + 1)
question_options_df["label"] = question_options_df.apply(
    lambda r: f"{int(r['display_order'])}. ({r['question_id']}) | {r['dimension']} | {r['question']}",
    axis=1
)

question_labels = question_options_df["label"].tolist()
question_id_options = sorted(
    question_options_df["question_id"].dropna().astype(str).unique().tolist(),
    key=numeric_question_sort_key
)

if "main_pipeline" not in st.session_state:
    st.session_state["main_pipeline"] = MainQuestionPipeline(top_k=8)

pipeline = st.session_state["main_pipeline"]


def make_progress_callback(progress_bar, status_box):
    def _callback(percent: int, message: str):
        percent = max(0, min(int(percent), 100))
        progress_bar.progress(percent, text=f"{percent}%")
        status_box.info(message)
    return _callback


def reset_history_page():
    st.session_state["history_page"] = 1


tab1, tab2 = st.tabs(["Run Question", "Saved Runs"])

with tab1:
    st.subheader("Run a Question")

    st.markdown("#### Step 1 — Select country")
    selected_country = st.selectbox("Country", demo_country_options, label_visibility="collapsed")

    st.markdown("#### Step 2 — Select ODM question")
    selected_question_label = st.selectbox(
        "ODM Question",
        question_labels,
        label_visibility="collapsed"
    )

    selected_question_row = question_options_df[
        question_options_df["label"] == selected_question_label
    ].iloc[0]

    selected_question_id = str(selected_question_row["question_id"]).strip()
    selected_question_text = str(selected_question_row["question"]).strip()

    st.info(f"Selected question: {selected_question_text}")

    if selected_country not in evaluated_countries:
        st.info(
            "This country is outside the frozen evaluation scope. "
            "The system will first check the prepared evidence base. "
            "If coverage is insufficient, it will dynamically run the same main pipeline steps "
            "from query generation through answer generation for this question."
        )

    if st.button("Run question"):
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        try:
            progress_bar = progress_placeholder.progress(0, text="0%")
            callback = make_progress_callback(progress_bar, status_placeholder)

            result = pipeline.run_question(
                country=selected_country,
                question_id=selected_question_id,
                progress_callback=callback,
            )

            print(
                f"[UI RUN] country={selected_country} | question_id={selected_question_id} | "
                f"mode={result.get('mode', '')} | "
                f"answer={result.get('final_answer_selected', '')} | "
                f"score={result.get('score_suggestion', '')} | "
                f"confidence={result.get('confidence', '')}"
            )

            st.session_state["latest_result"] = result

            if result.get("mode") != "unsupported":
                save_search_result(result)
                st.success("Result saved to history")
            else:
                st.warning("This request is not supported, so no result was saved.")

        except Exception as e:
            st.error(f"Pipeline failed: {e}")
        finally:
            progress_placeholder.empty()
            status_placeholder.empty()

    if "latest_result" in st.session_state:
        st.markdown("---")
        render_full_result(st.session_state["latest_result"], key_suffix="latest")



with tab2:
    st.subheader("Saved Runs")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        filter_country = st.selectbox(
            "Filter by country",
            ["All"] + demo_country_options,
            key="filter_country"
        )

    with filter_col2:
        filter_question_id = st.selectbox(
            "Filter by question ID",
            ["All"] + question_id_options,
            key="filter_question_id"
        )

    selected_filter_country = None if filter_country == "All" else filter_country
    selected_filter_question_id = None if filter_question_id == "All" else filter_question_id

    filtered_rows = list_saved_results_filtered(
        country=selected_filter_country,
        question_id=selected_filter_question_id,
        limit=5000,
    )

    render_saved_runs_rows(filtered_rows, container_height=460)

    if "delete_saved_id" in st.session_state and st.session_state["delete_saved_id"] is not None:
        delete_saved_result(st.session_state["delete_saved_id"])
        st.success(f"Deleted row #{st.session_state['delete_saved_id']}")
        st.session_state["delete_saved_id"] = None
        st.rerun()

    if "load_saved_id" in st.session_state and st.session_state["load_saved_id"] is not None:
        loaded_result = get_saved_result(st.session_state["load_saved_id"])
        if loaded_result:
            print(
                f"[OPEN SAVED RESULT] id={st.session_state['load_saved_id']} | "
                f"country={loaded_result.get('country', '')} | "
                f"question_id={loaded_result.get('question_id', '')} | "
                f"mode={loaded_result.get('mode', '')}"
            )
            st.session_state["saved_result_view"] = loaded_result
        st.session_state["load_saved_id"] = None
        st.rerun()

    if "saved_result_view" in st.session_state:
        st.markdown("---")
        st.markdown("### Opened Saved Result")
        render_full_result(st.session_state["saved_result_view"], key_suffix="saved")