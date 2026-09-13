from pathlib import Path
import hashlib
import re
import uuid
import argparse
import fitz
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


# --------------------------------------------------
# CONFIG
# --------------------------------------------------


# COLLECTION_NAME = "financial_documents"

# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "financial_documents_bge"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def clean_text(text: str) -> str:
    """
    Basic PDF text cleanup.
    """

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def build_chunk(blocks):
    """
    Convert PDF blocks into a single chunk.
    """

    text = "\n\n".join(
        block["text"]
        for block in blocks
    )

    return {
        "text": text,
        "block_count": len(blocks),
    }
def get_heading_level(text: str):
    """
    Detect structural heading level from common document patterns.

    Returns:
        1 = major section
        2 = numbered subsection/question
        3 = lettered subsection
        None = normal content
    """

    text = text.strip()

    if not text:
        return None

    # Major sections:
    # I. Overview
    # IV. Employees
    # V. Holding, Subsidiary and Associate Companies
    if re.match(
        r"^(I|II|III|IV|V|VI|VII|VIII|IX|X)[\.\)]\s+",
        text,
        re.IGNORECASE,
    ):
        return 1

    # Numbered sections/questions:
    # 19. Markets served...
    # 20. Details at the end...
    # 23. a Names of...
    if re.match(r"^\d+[\.\)]\s+", text):
        return 2

    # Lettered subsections:
    # a. Number of locations
    # b. Export contribution
    # c. A brief on types of customers
    if re.match(r"^[a-z][\.\)]\s+", text, re.IGNORECASE):
        return 3

    return None

def chunk_blocks(blocks, max_words=300, overlap_words=50):
    chunks = []

    current_blocks = []
    current_word_count = 0

    for block in blocks:
        text = block["text"]
        word_count = len(text.split())

        heading_level = get_heading_level(text)

        # Major section or numbered subsection:
        # start a new semantic chunk.
        if heading_level in (1, 2) and current_blocks:
            chunks.append(build_chunk(current_blocks))

            current_blocks = []
            current_word_count = 0

        # Split oversized chunks.
        if (
            current_blocks
            and current_word_count + word_count > max_words
        ):
            chunks.append(build_chunk(current_blocks))

            overlap_blocks = []
            overlap_count = 0

            for previous_block in reversed(current_blocks):
                previous_words = len(
                    previous_block["text"].split()
                )

                if overlap_count + previous_words > overlap_words:
                    break

                overlap_blocks.insert(0, previous_block)
                overlap_count += previous_words

            current_blocks = overlap_blocks
            current_word_count = overlap_count

        current_blocks.append(block)
        current_word_count += word_count

    if current_blocks:
        chunks.append(build_chunk(current_blocks))

    return chunks


def make_document_id(pdf_path: Path) -> str:
    """
    Create a stable ID based on the file contents.
    """

    sha256 = hashlib.sha256()

    with open(pdf_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()



def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest a financial PDF into Qdrant"
    )

    parser.add_argument(
        "pdf",
        type=Path,
        help="Path to the PDF file"
    )

    parser.add_argument(
        "--company",
        help="Company name, e.g. IRFC"
    )

    parser.add_argument(
        "--financial-year",
        type=int,
        help="Financial year, e.g. 2025"
    )

    parser.add_argument(
        "--document-type",
        default="annual_report",
        help="Document type, e.g. annual_report"
    )

    return parser.parse_args()

# --------------------------------------------------
# EXTRACT PDF
# --------------------------------------------------

def extract_pdf(pdf_path: Path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_number, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]

        page_blocks = []

        for block in blocks:
            if "lines" not in block:
                continue

            block_text_parts = []
            font_sizes = []
            font_names = []

            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()

                    if not text:
                        continue

                    block_text_parts.append(text)
                    font_sizes.append(span["size"])
                    font_names.append(span["font"])

            text = clean_text(" ".join(block_text_parts))

            if not text:
                continue

            page_blocks.append({
                "text": text,
                "x0": block["bbox"][0],
                "y0": block["bbox"][1],
                "x1": block["bbox"][2],
                "y1": block["bbox"][3],
                "font_size": max(font_sizes) if font_sizes else 0,
                "font_name": font_names[0] if font_names else "",
            })

        if page_blocks:
            pages.append({
                "page": page_number,
                "blocks": page_blocks,
            })

    doc.close()

    return pages



def debug_page(page):

    print()
    print("=" * 80)
    print(f"PAGE {page['page']}")
    print("=" * 80)


    for block_index, block in enumerate(page["blocks"]):
        print()
        print(f"BLOCK {block_index}")
        print(f"Position: ({block['x0']:.1f}, {block['y0']:.1f})")
        print(block["text"])


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    args = parse_args()

    pdf_path = args.pdf

    print(f"Reading: {pdf_path}")

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)


    document_id = make_document_id(pdf_path)

    print(f"Document ID: {document_id[:12]}...")

    # ----------------------------------------------
    # Extract
    # ----------------------------------------------

    pages = extract_pdf(pdf_path)


    debug_page(pages[95])

    print(f"Pages with text: {len(pages)}")

    # ----------------------------------------------
    # Chunk
    # ----------------------------------------------

    chunks = []

    for page in pages:

        page_chunks = chunk_blocks(
            page["blocks"],
            max_words=300,
            overlap_words=50
        )
        

        for chunk_index, chunk in enumerate(page_chunks):

            chunks.append({
                "document_id": document_id,

                "chunk_id": (
                    f"{args.company.lower()}"
                    f"-{args.financial_year}"
                    f"-p{page['page']}"
                    f"-c{chunk_index}"
                ),

                "page_start": page["page"],
                "page_end": page["page"],

                "chunk_index": chunk_index,

                "text": chunk["text"],

                "block_count": chunk["block_count"],
            })

    print(f"Total chunks: {len(chunks)}")

    # ----------------------------------------------
    # ADD DEBUG CODE HERE
    # ----------------------------------------------

    # print()
    # print("=" * 80)
    # print("CHUNK SAMPLE")
    # print("=" * 80)

    # for chunk in chunks:

    #     if chunk["page_start"] == 96:

    #         print()
    #         print(
    #             f"CHUNK {chunk['chunk_index']}"
    #         )

    #         print(
    #             chunk["text"]
    #         )

    #         print("-" * 80)


    # ----------------------------------------------
    # Embedding model
    # ----------------------------------------------

    print("Loading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        f"Embedding dimension: "
        f"{model.get_sentence_embedding_dimension()}"
    )

    # ----------------------------------------------
    # Generate embeddings
    # ----------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    print(
        f"Generated embeddings: "
        f"{embeddings.shape}"
    )

    # ----------------------------------------------
    # Qdrant
    # ----------------------------------------------

    client = QdrantClient(
        url="http://localhost:6333"
    )

    vector_size = embeddings.shape[1]

    # Create collection if necessary
    if not client.collection_exists(COLLECTION_NAME):

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )

        print(
            f"Created collection: {COLLECTION_NAME}"
        )

    # ----------------------------------------------
    # Upload
    # ----------------------------------------------

    points = []

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):

        # Stable UUID derived from chunk ID
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk["chunk_id"]
            )
        )

        points.append(
            models.PointStruct(
                id=point_id,

                vector=embedding.tolist(),

                payload={
                    "company": args.company,
                    "document_type": args.document_type,
                    "financial_year": args.financial_year,

                    "document_id": chunk["document_id"],
                    "chunk_id": chunk["chunk_id"],

                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],

                    "text": chunk["text"]
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )

    print()
    print("===================================")
    print("INGESTION COMPLETE")
    print("===================================")
    print(
        f"Document : "
        f"{args.company} "
        f"{args.document_type} "
        f"FY{args.financial_year}"
    )
    print(f"Pages    : {len(pages)}")
    print(f"Chunks   : {len(chunks)}")
    print(f"Vectors  : {len(points)}")
    print(f"Qdrant   : {COLLECTION_NAME}")


if __name__ == "__main__":
    main()