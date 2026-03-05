import json, argparse
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_chunks(chunks_path: Path):
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model_name", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    chunks = load_chunks(data_dir / "chunks.jsonl")

    index = faiss.read_index(str(data_dir / "faiss.index"))
    embedder = SentenceTransformer(args.model_name)
    q_emb = embedder.encode([args.q], normalize_embeddings=True).astype("float32")

    scores, idxs = index.search(q_emb, args.k)
    for rank, (i, s) in enumerate(zip(idxs[0], scores[0]), start=1):
        c = chunks[int(i)]
        print(f"\n#{rank} score={float(s):.4f} {c['company']} {c['year']} section={c['section']} chunk={c['chunk_index']}")
        print(c["text"][:600], "...")


if __name__ == "__main__":
    main()