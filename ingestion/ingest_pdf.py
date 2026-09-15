from pathlib import Path
import hashlib
import re
import uuid
import argparse
import pymupdf
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from metadata import (
    get_or_create_company,
    get_or_create_document,
    insert_chunk,
)


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
        description="Ingest financial PDFs for a company"
    )

    parser.add_argument(
        "--company",
        required=True,
        help="Company ticker/folder name, e.g. irfc"
    )

    parser.add_argument(
        "--document-type",
        default="annual_report",
        help="Document type, e.g. annual_report"
    )

    return parser.parse_args()

def find_company_pdfs(company: str, document_type: str):
    project_root = Path(__file__).resolve().parent.parent

    company_dir = project_root / "companies" / company.lower()

    if document_type == "annual_report":
        pdf_dir = company_dir / "annual-reports"
    else:
        raise ValueError(
            f"Unsupported document type: {document_type}"
        )

    if not pdf_dir.exists():
        raise FileNotFoundError(
            f"Document directory does not exist: {pdf_dir}"
        )

    pdfs = sorted(pdf_dir.glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(
            f"No PDF files found in: {pdf_dir}"
        )

    return pdfs

def extract_report_year(pdf_path: Path) -> int:
    """
    Extract report year from filename.

    Example:
        irfc-ar-2026.pdf -> 2026
    """

    match = re.search(r"(\d{4})", pdf_path.stem)

    if not match:
        raise ValueError(
            f"Could not determine report year from filename: {pdf_path.name}"
        )

    return int(match.group(1))

# --------------------------------------------------
# EXTRACT PDF
# --------------------------------------------------

def extract_pdf(pdf_path: Path):
    doc = pymupdf.open(pdf_path)
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


def process_document(pdf_path: Path, args, company_id: int):
    """
    Extract and structurally chunk a single financial document.
    """

    report_year = extract_report_year(pdf_path)
    content_hash = make_document_id(pdf_path)

    document_id = get_or_create_document(
        company_id=company_id,
        document_type=args.document_type,
        title=f"{args.company.upper()} Annual Report {report_year}",
        financial_year=None,
        quarter=None,
        document_date=None,
        file_path=str(pdf_path),
        page_count=None,
        content_hash=content_hash,
    )

    print()
    print("=" * 80)
    print(f"Processing : {pdf_path.name}")
    print(f"Report year: {report_year}")
    print(f"Document ID: {document_id}")
    print("=" * 80)

    # ----------------------------------------------
    # Extract
    # ----------------------------------------------

    pages = extract_pdf(pdf_path)

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
                    f"-{report_year}"
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
    # Store chunks in SQLite
    # ----------------------------------------------

    print("Storing chunks in SQLite...")

    for sequence_number, chunk in enumerate(chunks):

        insert_chunk(
            document_id=document_id,
            chunk_id=chunk["chunk_id"],
            page_start=chunk["page_start"],
            page_end=chunk["page_end"],
            sequence_number=sequence_number,
            text=chunk["text"],
            chunk_type="unknown",
        )

    print(f"Stored chunks in SQLite: {len(chunks)}")

    return {
        "document_id": document_id,
        "report_year": report_year,
        "content_hash": content_hash,
        "pages": pages,
        "chunks": chunks,
    }

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    args = parse_args()

    pdfs = find_company_pdfs(
        args.company,
        args.document_type
    )

    print()
    print("=" * 80)
    print(f"Company       : {args.company}")
    print(f"Document type : {args.document_type}")
    print(f"PDFs found    : {len(pdfs)}")
    print("=" * 80)

    # ----------------------------------------------
    # Company
    # ----------------------------------------------

    company_id = get_or_create_company(
        name=args.company.upper(),
        ticker=args.company.upper()
    )

    print(f"Company ID    : {company_id}")

    # ----------------------------------------------
    # Process documents
    # ----------------------------------------------

    for pdf_path in pdfs:

        result = process_document(
            pdf_path,
            args,
            company_id
        )

        print(
            f"Completed : {pdf_path.name} "
            f"| pages={len(result['pages'])} "
            f"| chunks={len(result['chunks'])}"
        )


if __name__ == "__main__":
    main()