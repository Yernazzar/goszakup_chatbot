"""
Semantic Search Pipeline using pgvector + sentence-transformers.
Embeds unique lot names from core_lots_cleaned and enables
cosine similarity search so queries like 'Канцелярия' find 'Степлер'.
"""
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

import os
from dotenv import load_dotenv
load_dotenv()

DB_CONFIG = {
    'dbname': os.getenv("DB_NAME", "goszakup_db"),
    'user': os.getenv("DB_USER", "postgres"),
    'password': os.getenv("DB_PASSWORD", "0000"),
    'host': os.getenv("DB_HOST", "localhost"),
    'port': os.getenv("DB_PORT", "5432")
}

# Multilingual model that handles Russian/Kazakh well, runs fully local, no API needed
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
EMBEDDING_DIM = 384  # This model outputs 384-dimensional vectors


def enable_pgvector(conn):
    """Try to enable the pgvector extension. Requires it to be installed server-side."""
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        print("pgvector extension enabled.")
        return True
    except Exception as e:
        conn.rollback()
        print(f"[WARNING] Could not enable pgvector: {e}")
        print("Falling back to float[] storage with FAISS-based search.")
        return False


def create_embeddings_table(conn, use_pgvector: bool):
    """Create the lot_embeddings table, using vector type if pgvector is available."""
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS lot_embeddings;")
    if use_pgvector:
        cur.execute(f"""
            CREATE TABLE lot_embeddings (
                clean_name  TEXT PRIMARY KEY,
                embedding   vector({EMBEDDING_DIM})
            );
        """)
        cur.execute("CREATE INDEX lot_emb_idx ON lot_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
    else:
        # Fallback: store as JSONB (array of floats) — Python-side FAISS handles search
        cur.execute("""
            CREATE TABLE lot_embeddings (
                clean_name  TEXT PRIMARY KEY,
                embedding   JSONB NOT NULL
            );
        """)
    conn.commit()
    print("lot_embeddings table created.")


def build_and_store_embeddings():
    print(f"Loading model '{MODEL_NAME}' (downloads once, then cached locally)...")
    model = SentenceTransformer(MODEL_NAME)

    conn = psycopg2.connect(**DB_CONFIG)
    use_pgvector = enable_pgvector(conn)
    if use_pgvector:
        register_vector(conn)

    create_embeddings_table(conn, use_pgvector)

    # Fetch all unique cleaned lot names
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT clean_name FROM core_lots_cleaned WHERE clean_name IS NOT NULL AND clean_name <> '' LIMIT 10000;")
    names = [row[0] for row in cur.fetchall()]
    print(f"Embedding {len(names)} unique lot names...")

    # Batch encode (fast on CPU, ~2-3 min for 8k names)
    BATCH_SIZE = 256
    all_embeddings = model.encode(names, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)

    # Insert into DB in batches
    import json
    insert_count = 0
    for i in range(0, len(names), BATCH_SIZE):
        batch_names = names[i:i+BATCH_SIZE]
        batch_vecs  = all_embeddings[i:i+BATCH_SIZE]
        for name, vec in zip(batch_names, batch_vecs):
            if use_pgvector:
                cur.execute(
                    "INSERT INTO lot_embeddings (clean_name, embedding) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
                    (name, vec.tolist())
                )
            else:
                cur.execute(
                    "INSERT INTO lot_embeddings (clean_name, embedding) VALUES (%s, %s::jsonb) ON CONFLICT DO NOTHING;",
                    (name, json.dumps(vec.tolist()))
                )
        conn.commit()
        insert_count += len(batch_names)
        print(f"  Stored {insert_count}/{len(names)}...")

    conn.close()
    print("Embedding pipeline completed!")


if __name__ == "__main__":
    build_and_store_embeddings()
