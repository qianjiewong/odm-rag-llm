from pathlib import Path

import pandas as pd

from config import PROCESSED_DIR
from step3_generate_queries import generate_queries_for_row, get_openai_client, get_search_language
from step4_search_web import search_web
from step5_download_docs import build_download_rows_for_question
from step6_extract_text import build_extracted_rows
from step7_chunk_text import build_chunk_rows
from step8_embed_chunks import embed_chunk_dataframe
from step9_retrieve_topk import retrieve_topk_for_question, get_retrieval_model
from step10_generate_answers import generate_answer_for_question


COUNTRY_NAME = "Estonia"
TOP_K = 8
MAX_URLS_PER_QUESTION = 8


def safe_text(value):
    return str(value or "").strip()


def get_response_scoring_value(row: pd.Series) -> str:
    """
    Safely retrieve response_scoring after merge.
    Handles cases like:
    - response_scoring
    - response_scoring_x
    - response_scoring_y
    """
    for col in ["response_scoring", "response_scoring_x", "response_scoring_y"]:
        if col in row.index:
            return safe_text(row.get(col, ""))
    return ""


def main():
    out_dir = Path("debug_estonia_rerun")
    out_dir.mkdir(exist_ok=True)

    gt_path = PROCESSED_DIR / "ground_truth_odm_selected.csv"
    registry_path = PROCESSED_DIR / "registry_odm_selected_questions.csv"

    gt_df = pd.read_csv(gt_path)
    registry_df = pd.read_csv(registry_path)

    gt_df["country"] = gt_df["country"].astype(str).str.strip()
    gt_df["question_id"] = gt_df["question_id"].astype(str).str.strip()
    registry_df["question_id"] = registry_df["question_id"].astype(str).str.strip()

    estonia_df = gt_df[gt_df["country"] == COUNTRY_NAME].copy()

    # only merge response_scoring if gt_df does not already contain it cleanly
    if "response_scoring" not in estonia_df.columns:
        estonia_df = estonia_df.merge(
            registry_df[["question_id", "response_scoring"]],
            on="question_id",
            how="left"
        )
    else:
        # if it exists but may have missing values, fill from registry
        registry_scoring = registry_df[["question_id", "response_scoring"]].copy()
        estonia_df = estonia_df.merge(
            registry_scoring,
            on="question_id",
            how="left",
            suffixes=("", "_registry")
        )

        if "response_scoring_registry" in estonia_df.columns:
            estonia_df["response_scoring"] = estonia_df["response_scoring"].fillna(
                estonia_df["response_scoring_registry"]
            )
            estonia_df = estonia_df.drop(columns=["response_scoring_registry"])

    print(f"Estonia question rows: {len(estonia_df)}")
    print(f"Columns available in estonia_df: {list(estonia_df.columns)}")

    # Step 3: queries
    query_client = get_openai_client()
    query_rows = []

    for _, row in estonia_df.iterrows():
        country = safe_text(row["country"])
        question_id = safe_text(row["question_id"])
        dimension = safe_text(row["dimension"])
        question = safe_text(row["question"])

        try:
            queries = generate_queries_for_row(
                client=query_client,
                country=country,
                question_id=question_id,
                question=question,
                dimension=dimension,
            )
        except Exception as e:
            print(f"Query generation failed for {country} | {question_id}: {e}")
            queries = []

        for i, query in enumerate(queries, start=1):
            query_rows.append({
                "country": country,
                "question_id": question_id,
                "dimension": dimension,
                "question": question,
                "query_no": i,
                "query": query,
                "search_language": get_search_language(country),
            })

    queries_df = pd.DataFrame(query_rows)
    queries_df.to_csv(out_dir / "queries_estonia.csv", index=False)
    print(f"Saved queries: {len(queries_df)}")

    # Step 4: search
    search_rows = []

    for _, row in queries_df.iterrows():
        country = safe_text(row["country"])
        question_id = safe_text(row["question_id"])
        dimension = safe_text(row["dimension"])
        question = safe_text(row["question"])
        query_no = row["query_no"]
        query = safe_text(row["query"])
        search_language = safe_text(row["search_language"])

        try:
            results = search_web(query=query, max_results=8, country=country)
        except Exception as e:
            print(f"Search failed for {country} | {question_id} | {query}: {e}")
            results = []

        for result in results:
            search_rows.append({
                "country": country,
                "question_id": question_id,
                "dimension": dimension,
                "question": question,
                "query_no": query_no,
                "query": query,
                "search_language": search_language,
                "rank": result["rank"],
                "title": result["title"],
                "url": result["url"],
                "snippet": result["snippet"],
                "published_date": result["published_date"],
                "provider": result["provider"],
            })

    search_df = pd.DataFrame(search_rows).drop_duplicates(subset=["country", "question_id", "query", "url"])
    search_df.to_csv(out_dir / "search_results_estonia.csv", index=False)
    print(f"Saved search results: {len(search_df)}")

    # Step 5: download
    download_rows = []

    for question_id, qdf in search_df.groupby("question_id"):
        question_out_dir = out_dir / "downloads" / question_id
        question_out_dir.mkdir(parents=True, exist_ok=True)

        rows, blocked_answer, blocked_procurement = build_download_rows_for_question(
            qdf=qdf,
            country=COUNTRY_NAME,
            question_id=question_id,
            base_out_dir=question_out_dir,
            max_urls_per_question=MAX_URLS_PER_QUESTION,
            use_slug_filenames=False,
            sleep_seconds=0.0,
            verbose=True,
        )

        print(
            f"{question_id}: downloaded={len(rows)} | "
            f"blocked_answer={blocked_answer} | blocked_procurement={blocked_procurement}"
        )

        download_rows.extend(rows)

    download_df = pd.DataFrame(download_rows)
    download_df.to_csv(out_dir / "download_index_estonia.csv", index=False)
    print(f"Saved downloaded docs: {len(download_df)}")

    # Step 6: extract
    extracted_rows = build_extracted_rows(
        df=download_df,
        include_query_metadata=True,
        verbose=True,
    )
    extracted_df = pd.DataFrame(extracted_rows)
    extracted_df.to_csv(out_dir / "extracted_text_estonia.csv", index=False)
    print(f"Saved extracted text: {len(extracted_df)}")

    # Step 7: chunk
    chunk_rows = build_chunk_rows(
        df=extracted_df,
        include_query_metadata=True,
        verbose=True,
    )
    chunk_df = pd.DataFrame(chunk_rows)
    chunk_df.to_csv(out_dir / "chunks_estonia.csv", index=False)
    print(f"Saved chunks: {len(chunk_df)}")

    # Step 8: embed
    embeddings_dir = out_dir / "embeddings"
    _, chunk_embeddings = embed_chunk_dataframe(
        chunk_df=chunk_df,
        output_dir=embeddings_dir,
        embeddings_filename="chunks_estonia.npy",
        metadata_filename="chunks_estonia_metadata.pkl",
        show_progress_bar=True,
        model=None,
    )
    print(f"Saved embeddings: {chunk_embeddings.shape}")

    # Step 9: retrieve
    retrieval_model = get_retrieval_model()
    retrieval_rows = []

    for _, row in estonia_df.iterrows():
        question_id = safe_text(row["question_id"])
        dimension = safe_text(row["dimension"])
        question = safe_text(row["question"])

        retrieved_df = retrieve_topk_for_question(
            country=COUNTRY_NAME,
            question_id=question_id,
            question_text=question,
            dimension=dimension,
            top_k=TOP_K,
            model=retrieval_model,
            chunk_df=chunk_df,
            chunk_embeddings=chunk_embeddings,
        )

        retrieval_rows.extend(retrieved_df.to_dict(orient="records"))

    retrieval_df = pd.DataFrame(retrieval_rows)
    retrieval_df.to_csv(out_dir / "retrieval_estonia.csv", index=False)
    print(f"Saved retrieval rows: {len(retrieval_df)}")

    # Step 10: generate
    gen_client = get_openai_client()
    generated_rows = []

    for _, row in estonia_df.iterrows():
        question_id = safe_text(row["question_id"])
        dimension = safe_text(row["dimension"])
        question = safe_text(row["question"])
        response_scoring = get_response_scoring_value(row)

        q_retrieved = retrieval_df[
            retrieval_df["question_id"].astype(str).str.strip() == question_id
        ].copy()

        result = generate_answer_for_question(
            country=COUNTRY_NAME,
            question_id=question_id,
            question_text=question,
            dimension=dimension,
            response_scoring=response_scoring,
            retrieved_df=q_retrieved,
            client=gen_client,
        )

        generated_rows.append({
            "country": COUNTRY_NAME,
            "question_id": question_id,
            "dimension": dimension,
            "question": question,
            "response_scoring": response_scoring,
            "final_answer_selected": result.get("final_answer_selected", ""),
            "score_suggestion": result.get("score_suggestion", None),
            "justification_generated": result.get("justification_generated", ""),
            "confidence": result.get("confidence", None),
            "evidence_sufficiency": result.get("evidence_sufficiency", ""),
        })

    generated_df = pd.DataFrame(generated_rows)
    generated_df.to_csv(out_dir / "generated_answers_estonia.csv", index=False)
    print(f"Saved generated answers: {len(generated_df)}")

    print("\nDone. Check this folder:")
    print(out_dir.resolve())


if __name__ == "__main__":
    main()