import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from sentence_transformers import SentenceTransformer

from config import INTERIM_DIR
from utils import print_header, ensure_dir


EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE = 32


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def get_embedding_model() -> SentenceTransformer:
    """
    Shared model loader for BOTH:
    - evaluation batch path
    - UI runtime path
    """
    return SentenceTransformer(EMBED_MODEL_NAME)


def prepare_chunk_texts(chunk_df: pd.DataFrame) -> list[str]:
    """
    Shared text preparation for BOTH:
    - evaluation batch path
    - UI runtime path
    """
    if "chunk_text" not in chunk_df.columns:
        raise ValueError("Chunk dataframe must contain a 'chunk_text' column.")

    texts = [safe_text(x) for x in chunk_df["chunk_text"].tolist()]

    if not any(texts):
        raise ValueError("All chunk_text values are empty.")

    return texts


def generate_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = BATCH_SIZE,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """
    Shared embedding core for BOTH:
    - evaluation batch path
    - UI runtime path
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    ).astype("float32")

    return embeddings


def save_embedding_artifacts(
    chunk_df: pd.DataFrame,
    embeddings: np.ndarray,
    embeddings_path: Path,
    metadata_path: Path,
) -> None:
    """
    Shared artifact-saving logic for BOTH:
    - evaluation batch path
    - UI runtime path
    """
    np.save(embeddings_path, embeddings)

    with open(metadata_path, "wb") as f:
        pickle.dump(chunk_df, f)


def embed_chunk_dataframe(
    chunk_df: pd.DataFrame,
    output_dir: Path = None,
    embeddings_filename: str = "runtime_embeddings.npy",
    metadata_filename: str = "runtime_chunk_metadata.pkl",
    show_progress_bar: bool = False,
    model: SentenceTransformer = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Shared high-level embedding function for BOTH:
    - evaluation batch path
    - UI runtime path
    """
    if chunk_df.empty:
        return chunk_df.copy(), np.array([])

    texts = prepare_chunk_texts(chunk_df)

    if model is None:
        model = get_embedding_model()

    embeddings = generate_embeddings(
        model=model,
        texts=texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=show_progress_bar,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        embeddings_path = output_dir / embeddings_filename
        metadata_path = output_dir / metadata_filename
        save_embedding_artifacts(
            chunk_df=chunk_df,
            embeddings=embeddings,
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
        )

    return chunk_df.copy(), embeddings


def embed_chunks_for_question(
    country: str,
    question_id: str,
    chunk_df: pd.DataFrame,
    output_dir: Path = None,
):
    """
    Reusable single-question embedding for UI runtime path.
    Now uses the SAME shared embedding logic as the batch evaluation path.
    """
    return embed_chunk_dataframe(
        chunk_df=chunk_df,
        output_dir=output_dir,
        embeddings_filename="runtime_embeddings.npy",
        metadata_filename="runtime_chunk_metadata.pkl",
        show_progress_bar=False,
        model=None,
    )


def main():
    print_header("STEP 8: EMBED CHUNKS")

    input_path = INTERIM_DIR / "chunks" / "chunks_odm_selected.csv"
    output_dir = INTERIM_DIR / "embeddings"
    embeddings_path = output_dir / "chunks_odm_selected.npy"
    metadata_path = output_dir / "chunks_odm_selected_metadata.pkl"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {input_path}\n"
            f"Please run step7_chunk_text.py first."
        )

    ensure_dir(output_dir)

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError(f"Chunk file is empty: {input_path}")

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = get_embedding_model()

    print(f"Embedding {len(df)} chunks...")
    _, embeddings = embed_chunk_dataframe(
        chunk_df=df,
        output_dir=output_dir,
        embeddings_filename="chunks_odm_selected.npy",
        metadata_filename="chunks_odm_selected_metadata.pkl",
        show_progress_bar=True,
        model=model,
    )

    print(f"\nSaved embeddings to: {embeddings_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Embedding shape: {embeddings.shape}")

    print("\nPreview:")
    preview_df = df[[
        "country", "question_id", "chunk_index", "chunk_length_words", "title"
    ]].head(10)
    print(preview_df.to_string(index=False))


if __name__ == "__main__":
    main()