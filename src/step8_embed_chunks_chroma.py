import numpy as np
import pandas as pd
from pathlib import Path
import pickle

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import INTERIM_DIR
from utils import print_header, ensure_dir


EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE = 32

CHROMA_DIRNAME = "chroma_db"
CHROMA_COLLECTION_NAME = "odm_chunks_selected"


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def get_embedding_model() -> SentenceTransformer:
    """
    Shared embedding model loader.
    Uses exactly the same model as the current file-based Step 8,
    so the Chroma experiment stays comparable.
    """
    return SentenceTransformer(EMBED_MODEL_NAME)


def prepare_chunk_texts(chunk_df: pd.DataFrame) -> list[str]:
    """
    Shared text preparation.
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
    Shared embedding generation.
    Produces the same normalized float32 vectors as your current Step 8.
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
    Optional artifact saving so this experiment still keeps .npy / .pkl outputs
    for inspection and comparison.
    """
    np.save(embeddings_path, embeddings)

    with open(metadata_path, "wb") as f:
        pickle.dump(chunk_df, f)


def get_chroma_client(chroma_dir: Path):
    """
    Create a persistent local ChromaDB client.
    This stores the vector DB on disk inside data/interim/chroma_db/.
    """
    chroma_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    return client


def recreate_collection(client, collection_name: str):
    """
    For a clean experiment run, delete the old collection if it exists
    and recreate it from scratch.
    """
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "ODM selected chunk embeddings"},
    )
    return collection


def build_chroma_records(chunk_df: pd.DataFrame, embeddings: np.ndarray):
    """
    Convert dataframe rows into Chroma insert format.

    Chroma stores:
    - ids
    - embeddings
    - documents
    - metadatas

    We keep metadata fields that will be useful later in retrieval reconstruction.
    """
    ids = []
    documents = []
    metadatas = []

    for idx, row in chunk_df.reset_index(drop=True).iterrows():
        chunk_id = safe_text(row.get("chunk_id", ""))
        if not chunk_id:
            chunk_id = f"chunk_{idx}"

        # Make ids globally unique and stable for Chroma
        chroma_id = f"{safe_text(row.get('country', 'unknown'))}__{safe_text(row.get('question_id', 'unknown'))}__{chunk_id}__{idx}"

        metadata = {
            "country": safe_text(row.get("country", "")),
            "question_id": safe_text(row.get("question_id", "")),
            "dimension": safe_text(row.get("dimension", "")),
            "question": safe_text(row.get("question", "")),
            "title": safe_text(row.get("title", "")),
            "url": safe_text(row.get("url", "")),
            "saved_path": safe_text(row.get("saved_path", "")),
            "file_type": safe_text(row.get("file_type", "")),
            "chunk_id": chunk_id,
            "chunk_index": int(row.get("chunk_index", 0)) if str(row.get("chunk_index", "")).strip() != "" else 0,
            "chunk_length_words": int(row.get("chunk_length_words", 0)) if str(row.get("chunk_length_words", "")).strip() != "" else 0,
        }

        ids.append(chroma_id)
        documents.append(safe_text(row.get("chunk_text", "")))
        metadatas.append(metadata)

    embedding_list = embeddings.tolist()

    return ids, documents, metadatas, embedding_list


def add_records_in_batches(
    collection,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    embeddings: list[list[float]],
    batch_size: int = 1000,
) -> None:
    """
    Insert records into Chroma in batches.
    This is safer for large chunk sets than inserting everything at once.
    """
    total = len(ids)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )

        print(f"Inserted into Chroma: {end}/{total}")


def embed_chunk_dataframe_to_chroma(
    chunk_df: pd.DataFrame,
    output_dir: Path = None,
    embeddings_filename: str = "chunks_odm_selected_chroma.npy",
    metadata_filename: str = "chunks_odm_selected_chroma_metadata.pkl",
    show_progress_bar: bool = False,
    model: SentenceTransformer = None,
    chroma_dir: Path = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    High-level Chroma embedding pipeline:
    1. prepare texts
    2. generate embeddings
    3. optionally save .npy/.pkl artifacts
    4. store embeddings + metadata in ChromaDB
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

    if chroma_dir is None:
        raise ValueError("chroma_dir must be provided.")

    client = get_chroma_client(chroma_dir)
    collection = recreate_collection(client, collection_name)

    ids, documents, metadatas, embedding_list = build_chroma_records(chunk_df, embeddings)

    add_records_in_batches(
        collection=collection,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embedding_list,
        batch_size=1000,
    )

    print(f"Chroma collection '{collection_name}' now contains {collection.count()} records.")

    return chunk_df.copy(), embeddings


def main():
    print_header("STEP 8 EXPERIMENT: EMBED CHUNKS TO CHROMADB")

    input_path = INTERIM_DIR / "chunks" / "chunks_odm_selected.csv"
    output_dir = INTERIM_DIR / "embeddings"
    chroma_dir = INTERIM_DIR / CHROMA_DIRNAME

    embeddings_path = output_dir / "chunks_odm_selected_chroma.npy"
    metadata_path = output_dir / "chunks_odm_selected_chroma_metadata.pkl"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {input_path}\n"
            f"Please run step7_chunk_text.py first."
        )

    ensure_dir(output_dir)
    ensure_dir(chroma_dir)

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError(f"Chunk file is empty: {input_path}")

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = get_embedding_model()

    print(f"Embedding {len(df)} chunks and storing in ChromaDB...")
    _, embeddings = embed_chunk_dataframe_to_chroma(
        chunk_df=df,
        output_dir=output_dir,
        embeddings_filename="chunks_odm_selected_chroma.npy",
        metadata_filename="chunks_odm_selected_chroma_metadata.pkl",
        show_progress_bar=True,
        model=model,
        chroma_dir=chroma_dir,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    print(f"\nSaved experimental embeddings to: {embeddings_path}")
    print(f"Saved experimental metadata to: {metadata_path}")
    print(f"Chroma database directory: {chroma_dir}")
    print(f"Embedding shape: {embeddings.shape}")

    print("\nPreview:")
    preview_df = df[
        ["country", "question_id", "chunk_index", "chunk_length_words", "title"]
    ].head(10)
    print(preview_df.to_string(index=False))


if __name__ == "__main__":
    main()