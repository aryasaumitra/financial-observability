import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "financial_observability.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    return sqlite3.connect(DB_PATH)


def create_tables():
    conn = get_connection()

    cursor = conn.cursor()

    # ----------------------------------------------
    # Companies
    # ----------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            ticker TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ----------------------------------------------
    # Documents
    # ----------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,

            document_type TEXT NOT NULL,
            title TEXT,

            financial_year TEXT,
            quarter TEXT,
            document_date TEXT,

            file_path TEXT,
            page_count INTEGER,

            content_hash TEXT,

            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (company_id)
                REFERENCES companies(id)
        )
    """)

    # ----------------------------------------------
    # Chunks
    # ----------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            document_id INTEGER NOT NULL,

            chunk_id TEXT NOT NULL,

            page_start INTEGER,
            page_end INTEGER,

            sequence_number INTEGER,

            section TEXT,
            subsection TEXT,

            chunk_type TEXT,

            text TEXT NOT NULL,
            word_count INTEGER,

            speaker TEXT,
            speaker_role TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (document_id)
                REFERENCES documents(id)
        )
    """)

    conn.commit()
    conn.close()



def get_or_create_company(name, ticker=None):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id
        FROM companies
        WHERE ticker = ?
        """,
        (ticker,)
    )

    row = cursor.fetchone()

    if row:
        conn.close()
        return row[0]

    cursor.execute(
        """
        INSERT INTO companies (name, ticker)
        VALUES (?, ?)
        """,
        (name, ticker)
    )

    company_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return company_id


def create_document(
    company_id,
    document_type,
    title,
    financial_year=None,
    quarter=None,
    document_date=None,
    file_path=None,
    page_count=None,
    content_hash=None,
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO documents (
            company_id,
            document_type,
            title,
            financial_year,
            quarter,
            document_date,
            file_path,
            page_count,
            content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            company_id,
            document_type,
            title,
            financial_year,
            quarter,
            document_date,
            file_path,
            page_count,
            content_hash,
        )
    )

    document_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return document_id


def get_or_create_document(
    company_id,
    document_type,
    title,
    financial_year=None,
    quarter=None,
    document_date=None,
    file_path=None,
    page_count=None,
    content_hash=None,
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id
        FROM documents
        WHERE company_id = ?
          AND document_type = ?
          AND financial_year IS ?
          AND quarter IS ?
          AND file_path = ?
        """,
        (
            company_id,
            document_type,
            financial_year,
            quarter,
            file_path,
        )
    )

    row = cursor.fetchone()

    if row:
        conn.close()
        return row[0]

    cursor.execute(
        """
        INSERT INTO documents (
            company_id,
            document_type,
            title,
            financial_year,
            quarter,
            document_date,
            file_path,
            page_count,
            content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            company_id,
            document_type,
            title,
            financial_year,
            quarter,
            document_date,
            file_path,
            page_count,
            content_hash,
        )
    )

    document_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return document_id

def insert_chunk(
    document_id,
    chunk_id,
    page_start,
    page_end,
    sequence_number,
    text,
    chunk_type="unknown",
):
    conn = get_connection()
    cursor = conn.cursor()

    # Check whether this chunk already exists
    cursor.execute(
        """
        SELECT id
        FROM chunks
        WHERE chunk_id = ?
        """,
        (chunk_id,)
    )

    row = cursor.fetchone()

    if row:
        conn.close()
        return row[0]

    cursor.execute(
        """
        INSERT INTO chunks (
            document_id,
            chunk_id,
            page_start,
            page_end,
            sequence_number,
            chunk_type,
            text,
            word_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            document_id,
            chunk_id,
            page_start,
            page_end,
            sequence_number,
            chunk_type,
            text,
            len(text.split()),
        )
    )

    chunk_db_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return chunk_db_id

def update_document_status(document_id, status):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE documents
        SET ingestion_status = ?
        WHERE id = ?
        """,
        (status, document_id)
    )

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_tables()

    print(f"Database created at: {DB_PATH}")