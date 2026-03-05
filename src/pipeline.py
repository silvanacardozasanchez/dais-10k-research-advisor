import os, re, json, argparse, hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from tqdm import tqdm

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# --------- helpers ---------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_text_pymupdf(pdf_path: Path) -> Dict[str, Any]:
    doc = fitz.open(str(pdf_path))
    pages = []
    for i in range(len(doc)):
        t = doc.load_page(i).get_text("text")
        pages.append({"page": i + 1, "text": clean_text(t)})
    full_text = "\n\n".join([p["text"] for p in pages if p["text"]])
    return {"pages": pages, "full_text": clean_text(full_text), "num_pages": len(doc)}

def parse_company_year(filename: str) -> Dict[str, Optional[str]]:
    # Adjust this to match how you name files, e.g. "Tesla_2023_10K.pdf"
    base = Path(filename).stem
    m = re.match(r"(?P<company>.+?)_(?P<year>\d{4})", base)
    if not m:
        return {"company": None, "year": None}
    return {"company": m.group("company").replace("-", " "), "year": m.group("year")}

def extract_risk_factors(text: str) -> str:
    """
    Very lightweight heuristic:
    grabs content between "Item 1A" and "Item 1B" (common in 10-K).
    If not found, returns full text.
    """
    t = text
    # make matching more robust
    t_norm = re.sub(r"\s+", " ", t).lower()
    start = t_norm.find("item 1a")
    if start == -1:
        return text
    end = t_norm.find("item 1b", start + 6)
    if end == -1:
        end = min(len(t), start + 200000)  # fallback window
    # map normalized positions back approximately by slicing original text
    # (good enough for milestone 2; you can refine later)
    return t[start:end]

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple character-based chunking. For Milestone 2 this is perfectly acceptable.
    Later you can switch to token-based chunking.
    """
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks


# --------- main pipeline ---------
def run_pipeline(pdf_dir: Path, out_dir: Path, model_name: str,
                 chunk_size: int, overlap: int, risk_only: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = out_dir / "chunks.jsonl"
    docs_path = out_dir / "docs.jsonl"
    id_map_path = out_dir / "id_map.json"
    faiss_path = out_dir / "faiss.index"

    all_chunks: List[Dict[str, Any]] = []
    all_docs: List[Dict[str, Any]] = []

    pdf_files = sorted([p for p in pdf_dir.glob("*.pdf")])
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    # 1) Ingest + extract + (optional) filter + chunk + metadata
    for pdf in tqdm(pdf_files, desc="Processing PDFs"):
        meta = parse_company_year(pdf.name)
        extracted = extract_text_pymupdf(pdf)

        source_text = extracted["full_text"]
        section = "full_document"

        if risk_only:
            source_text = extract_risk_factors(source_text)
            section = "item_1a_risk_factors"

        chunks = chunk_text(source_text, chunk_size=chunk_size, overlap=overlap)

        doc_id = sha1(str(pdf.resolve()))
        doc_record = {
            "doc_id": doc_id,
            "file_name": pdf.name,
            "company": meta["company"],
            "year": meta["year"],
            "num_pages": extracted["num_pages"],
            "section_mode": section,
            "num_chunks": len(chunks),
        }
        all_docs.append(doc_record)

        for idx, c in enumerate(chunks):
            chunk_id = sha1(doc_id + f"::{idx}")
            all_chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "company": meta["company"],
                "year": meta["year"],
                "section": section,
                "chunk_index": idx,
                "text": c
            })

    # Write doc-level + chunk-level storage (counts for M2 “written to storage”)
    with open(docs_path, "w", encoding="utf-8") as f:
        for r in all_docs:
            f.write(json.dumps(r) + "\n")

    with open(chunks_path, "w", encoding="utf-8") as f:
        for r in all_chunks:
            f.write(json.dumps(r) + "\n")

    # 2) Embeddings + vector index
    embedder = SentenceTransformer(model_name)
    texts = [c["text"] for c in all_chunks]
    emb = embedder.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine-like with normalized vectors
    index.add(emb)

    faiss.write_index(index, str(faiss_path))

    # 3) id_map (vector row -> chunk_id)
    id_map = {str(i): all_chunks[i]["chunk_id"] for i in range(len(all_chunks))}
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2)

    print("\nPipeline complete.")
    print(f"Docs written:   {docs_path}")
    print(f"Chunks written: {chunks_path}")
    print(f"FAISS index:    {faiss_path}")
    print(f"Chunks indexed: {len(all_chunks)}")
    print("Example doc record:", all_docs[0] if all_docs else None)
    print("Example chunk record:", all_chunks[0] if all_chunks else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--chunk_size", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--risk_only", action="store_true",
                    help="If set, try to isolate Item 1A Risk Factors before chunking")
    args = ap.parse_args()

    run_pipeline(
        pdf_dir=Path(args.pdf_dir),
        out_dir=Path(args.out_dir),
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        risk_only=args.risk_only
    )

if __name__ == "__main__":
    main()