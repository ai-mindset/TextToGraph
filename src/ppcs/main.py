"""Process and store document data, including embeddings and relationships, in an SQLite database, using LangChain. This module includes functions to initialise the database with required tables, save documents and their processed data, and save character and relationship data. Each function handles specific tasks such as validating input, interacting with the database using SQLAlchemy, and parsing extracted text into structured data. Logging is included for better visibility of its operations."""

# %%
import io
import json
import re
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict
from logging import Logger
from pathlib import Path
from re import Pattern

import ipdb
import numpy as np
import numpy.typing as npt
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import Client
from pydantic import BaseModel, Field

from ppcs.constants import Constants
from ppcs.logger import setup_logger

# %%
# Constants instance
constants = Constants()

logger: Logger = setup_logger(constants.LOG_LEVEL)


# %% ✅
class Character(BaseModel):
    """Character model with traits and relationships."""

    id: str = Field(default_factory=str)
    traits: list[str] = Field(default_factory=list[str])


class Relationship(BaseModel):
    """Relationship between two characters."""

    source: str = Field(default_factory=str)
    target: str = Field(default_factory=str)
    relationship: str = Field(default_factory=str)
    weight: float = Field(ge=0.0, le=1.0, default_factory=float)


# %% ✅
@contextmanager
def get_db_connection(db_path: str) -> Generator:
    """Create and manage database connection with proper error handling.

    Args:
        db_path: Path to SQLite database

    Yields:
        SQLite connection object

    Examples:
        >>> from ppcs.constants import Constants
        >>> constants =
        >>> db_path = "data/world.db"
        >>> with get_db_connection(db_path=db_path) as conn:
        ...     isinstance(conn, sqlite3.Connection)
        True

    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()  # Auto-commit successful transactions
    except Exception:
        if conn:
            conn.rollback()  # Rollback on error
        raise
    finally:
        if conn is not None:
            conn.close()


# %% ✅
def read_file(file_path: Path) -> str:
    """Read document content from a text file with error handling.

    Args:
        file_path: Path to the document file

    Returns:
        The content of the document as a string

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file can't be accessed

    Examples:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(mode='w') as f:
        ...     _ = f.write("Test content")
        ...     f.flush()
        ...     content = read_file(Path(f.name))
        >>> content
        'Test content'

    """
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")
    try:
        return file_path.read_text(encoding="utf-8")
    except PermissionError as e:
        raise PermissionError(f"Cannot access file {file_path}: {e}")


# %% ✅
def split_text(
    text: str,
    chunk_size: int = constants.CHUNK_SIZE,
    overlap: int = constants.CHUNK_OVERLAP,
    separators: list[str] = constants.SEPARATORS,
) -> list[str]:
    """Split text into chunks using RecursiveCharacterTextSplitter.

    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        separators: Tuple of separators to use, in order of preference

    Returns:
        List of text chunks

    Examples:
        >>> text = "This is a test. " * 500
        >>> from langchain_text_splitters import RecursiveCharacterTextSplitter
        >>> separators = ['\\n\\n', '\\n', '.', '!', '?', ',', ' ', '']
        >>> splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, separators=separators)
        >>> chunks = splitter.split_text(text)
        >>> len(chunks) > 0 and all(len(c) <= 100 for c in chunks)
        True

    """
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separators,
    )

    return splitter.split_text(text)


# %% ✅
def embed_text_chunks(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts using Ollama's nomic-embed-text model.

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors

    Examples:
        >>> texts = ["Test embedding 1", "Test embedding 2"]
        >>> embeddings = embed_text_batch(texts)
        >>> len(embeddings) == len(texts)
        True
    """
    if not texts:
        return []

    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return embeddings.embed_documents(texts)
    except Exception as e:
        raise RuntimeError(f"Embedding generation failed: {e}")


# %% ✅
def parse_line(
    line: str,
) -> tuple[Character, Character, Relationship]:
    """Parse a single line containing relationship information.

    Args:
        line: A line of text containing relationship information

    Returns:
        A tuple of (Character | None, Character | None, Relationship | None)

    >>> line = "Alice [traits: caring, empathetic] -> Related to -> Bob [traits: trusting, dependent] [strength: 0.9]."
    >>> char1, char2, rel = parse_line(line)
    >>> rel.source, rel.target, rel.weight
    ('Alice', 'Bob', 0.9)
    >>> line = "Not a relationship line"
    >>> parse_line(line)
    (Character(id='', traits=[]), Character(id='', traits=[]), Relationship(source='', target='', relationship='', weight=0.0))

    """

    relationship_pattern: Pattern[str] = re.compile(
        (
            r"(\w+)"  # Character1 name (source)
            r"\s*\[traits:\s*([^\]]+)\]?"  # Optional traits for source: matches "traits: x, y" inside brackets
            r"\s*->\s*"  # First arrow with optional whitespace
            r"([^->]+?)"  # Relationship description (non-greedy match)
            r"\s*->\s*"  # Second arrow with optional whitespace
            r"(\w+)"  # Character2 name (target)
            r"\s*\[traits:\s*([^\]]+)\]?"  # Optional traits for target: matches "traits: x, y" inside brackets
            r"\s*\[strength:\s*([\d.]+)\]"  # Strength value: matches decimal number inside [strength: X.X]
        ),
        re.VERBOSE,
    )

    match = re.match(relationship_pattern, line)

    char1 = Character()
    char2 = Character()
    rel = Relationship()

    if not match:
        return char1, char2, rel

    source, source_traits, relationship, target, target_traits, strength = match.groups()  # type: ignore[attr-defined]

    if source:
        char1 = Character(id=source, traits=[source_traits])

    if target:
        char2 = Character(id=target, traits=[target_traits])

    rel = Relationship(
        source=source,
        target=target,
        relationship=relationship,
        weight=float(strength),
    )

    return char1, char2, rel


# %% ✅
def is_valid_relationship_line(line: str) -> bool:
    """Check if a line contains a valid relationship format.

    Args:
        line: Line to validate

    Returns:
        bool: True if line matches expected format

    Examples:
        >>> line = "alice [traits: caring, empathetic] -> related to -> bob [traits: trusting, dependent] [strength: 0.9]."
        >>> is_valid_relationship_line(line)
        True
        >>> is_valid_relationship_line("Random text")
        False
    """
    # Normalize line for comparison
    normalised = line.lower().strip()

    # Check structural elements that must be present
    required_elements = [
        "traits:",  # Must have traits section
        "->",  # Must have relationship arrow
        "strength:",  # Must have strength indicator
    ]

    return all(element in normalised for element in required_elements)


# %% ✅
def extract_relationships(
    text: str,
) -> dict[int, list[dict]]:
    """Extract relationships from a given text using an LLM.

    Args:
        text (str): The text to analyze for character relationships.

    Returns:
        dict[int, list[dict]]: A dictionary where keys are line numbers and values are lists containing dictionaries of characters and their relationships. Each dictionary in the list contains 'character1', 'character2', and 'relationship' keys with corresponding values.

    ⚠️ this test may fail due to the stochastic nature of LLMs
    Examples:
        >>> text = "Alice is Bob's sister. They are very close. John is Mary's best friend. They've known each other for 20 years"
        >>> results = extract_relationships(text)
        >>> all(isinstance(item, dict) for item in results[1])
        True
        >>> any(r.get('source') == 'alice' and r.get('target') == 'bob' and r.get('relationship') == 'sibling' for r in results[1])
        True
        >>> alice_entry = next((item for item in results[1] if item.get('id') == 'alice'), None)
        >>> isinstance(alice_entry, dict) and 'traits' in alice_entry
        True
        >>> bob_entry = next((item for item in results[1] if item.get('id') == 'bob'), None)
        >>> isinstance(bob_entry, dict) and isinstance(bob_entry.get('traits'), list)
        True

    """
    try:
        llm = OllamaLLM(model="mistral", temperature=0.0)
        system_prompt = """Extract characters, character traits, relationships with other characters and each relationship strength from the following text. Use common terms such as 'sibling', 'spouse', 'friend','colleague', 'related to', 'depends on', 'influences', etc., for relationships. Traits should be presented in a list that starts with the word 'traits:' like so '[traits: ...]'. Traits could be 'strength', 'stamina', 'resourceful' etc. Only include traits if the text mentions some traits, otherwise generate an empty list like so '[traits:]'. Estimate a relationship's strength between 0.0 (very weak) and 1.0 (very strong). 
        Use this exact format: 
        Name of character1 [traits: a, b, ...] -> Relationship -> name of character2 [traits: c, d, ...] [strength: X.X].
        Do not include any other text in your response. 
        """

        logger.info("Invoking LLM...")
        response = llm.invoke(f"{system_prompt}\n\nText: {text}")

        parsed_results: dict[int, list] = {}

        character1 = Character()
        character2 = Character()
        relationship = Relationship()

        for i, line in enumerate(response.split("\n"), 1):
            line = line.strip().lower()
            is_valid = is_valid_relationship_line(line)
            if not line or not is_valid:
                parsed_results[i] = ["Skipped invalid line: '{line}'"]
                continue

            try:
                logger.info(f"Parsing... {line}\b")
                character1, character2, relationship = parse_line(line)
                parsed_results[i] = [
                    character1.model_dump(),
                    character2.model_dump(),
                    relationship.model_dump(),
                ]
            except ValueError as e:
                parsed_results[i] = ["Line {i}: {str(e)}"]
                raise

        return parsed_results
    except Exception as e:
        raise RuntimeError(f"Relationship extraction failed: {e}")


# %% ✅
def init_database(db_path: str) -> None:
    """Initialize SQLite database with required tables and indices.

    Args:
        db_path: Path to SQLite database

    Returns:
        None

    Examples:
        >>> import tempfile
        >>> db_path = tempfile.mkstemp()[1]
        >>> init_database(db_path)
        >>> # Check if the database is initialized correctly
        >>> conn = sqlite3.connect(db_path)
        >>> cursor = conn.cursor()
        >>> result = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        >>> isinstance(result, sqlite3.Cursor)
        True
        >>> tables = cursor.fetchall()
        >>> assert set(t[0] for t in tables) == {'documents', 'nodes', 'edges'}

    """
    with get_db_connection(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                chunks BLOB NOT NULL,
                embeddings BLOB NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                properties TEXT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS edges (
                source TEXT,
                target TEXT,
                relationship TEXT,
                weight REAL,
                PRIMARY KEY (source, target, relationship),
                FOREIGN KEY (source) REFERENCES nodes(id),
                FOREIGN KEY (target) REFERENCES nodes(id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
        """)


# %% ✅
def save_document(
    db_path: str,
    doc_id: str,
    content: str,
    chunks: list[str],
    embeddings: npt.NDArray[np.float32],
) -> None:
    """Save document and its processed data to database.

    Args:
        db_path: Path to SQLite database
        doc_id: Document identifier
        content: Original document content
        chunks: List of text chunks
        embeddings: Array of embeddings for each chunk

    Raises:
        ValueError: If `doc_id` or `content` is empty.
        ValueError: If the number of `chunks` does not match the number of `embeddings`.

    Examples:
        >>> db_path = "data/world.db"
        >>> init_database(db_path=db_path)
        >>> save_document(db_path=db_path, doc_id="doc123", content="This is a document.", chunks=["chunk1", "chunk2"], embeddings=np.array([[0.1, 0.2], [0.3, 0.4]]))
        >>> save_document(db_path=db_path, doc_id="", content="This should raise an error.", chunks=["chunk1", "chunk2"], embeddings=np.array([[0.1, 0.2], [0.3, 0.4]]))
        Traceback (most recent call last):
        ...
        ValueError: Document ID and content are required
        >>> save_document(db_path=db_path, doc_id="doc123", content="This should raise an error.", chunks=["chunk1"], embeddings=np.array([[0.1, 0.2], [0.3, 0.4]]))
        Traceback (most recent call last):
        ...
        ValueError: Number of chunks and embeddings must match

    """
    if not doc_id or not content:
        raise ValueError("Document ID and content are required")

    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks and embeddings must match")

    # Convert chunks to JSON string
    chunks_json = json.dumps(chunks)

    # Convert embeddings to binary using numpy
    embeddings_binary = io.BytesIO()
    np.save(embeddings_binary, embeddings)
    embeddings_binary_data = embeddings_binary.getvalue()

    with get_db_connection(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents (id, content, chunks, embeddings) VALUES (?, ?, ?, ?)",
            (
                doc_id,
                content,
                chunks_json,
                embeddings_binary_data,
            ),
        )
        conn.commit()


# ------------------------- Bookmark ---------------------------------------
# %% ✅
def save_graph_data(
    db_path: str, characters: list[Character], relationships: list[Relationship]
) -> None:
    """Save character and relationship data to a SQLite database.

    Args:
        db_path: Path to the SQLite database file.
        characters: List of Character objects representing the characters.
        relationships: List of Relationship objects representing the relationships between characters.

    Returns:
        None

    Example:
        >>> char1 = Character(id="C001", properties={"name": "John", "age": 30})
        >>> char2 = Character(id="C002", properties={"name": "Jane", "age": 25})
        >>> rel = Relationship(source="C001", target="C002", relationship="friend", weight=0.8)
        >>> db_path = "data/world.db"
        >>> save_graph_data(db_path=db_path, characters=[char1, char2], relationships=[rel])

    """
    if not characters and not relationships:
        return

    with get_db_connection(db_path) as conn:
        # Save unique characters
        unique_chars = {char.id: char for char in characters}

        for character in unique_chars.values():
            conn.execute(
                "INSERT OR REPLACE INTO nodes (id, properties) VALUES (?, ?)",
                (character.id, character.model_dump_json()),
            )

        # Save relationships
        for rel in relationships:
            conn.execute(
                "INSERT OR REPLACE INTO edges (source, target, relationship, weight) VALUES (?, ?, ?, ?)",
                (rel.source, rel.target, rel.relationship, rel.weight),
            )


# %% ✅
def get_document(db_path: str, doc_id: str):
    """
    Retrieve document from database.

    Args:
        db_path (str): Path to the database file.
        doc_id (str): ID of the document to retrieve.

    Returns:
        dict or None: A dictionary containing the document details if found, otherwise None.

    Examples:
        >>> db_path = "data/world.db"
        >>> init_database(db_path)
        >>> save_document(db_path=db_path, doc_id="doc123", content="This is a document.", chunks=["chunk1", "chunk2"], embeddings=np.array([[0.1, 0.2], [0.3, 0.4]]))
        >>> get_document(db_path=db_path, doc_id="doc123")
        {'id': 'doc123', 'content': 'This is a document.', 'chunks': ['chunk1', 'chunk2'], 'embeddings': array([[0.1, 0.2],
               [0.3, 0.4]])}

    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT content, chunks, embeddings FROM documents WHERE id = ?", (doc_id,)
        )
        row = cursor.fetchone()

        if row:
            content, chunks_json, embeddings_binary = row

            # Parse chunks from JSON
            chunks = json.loads(chunks_json)

            # Load embeddings from binary
            embeddings_io = io.BytesIO(embeddings_binary)
            embeddings = np.load(embeddings_io)

            return {
                "id": doc_id,
                "content": content,
                "chunks": chunks,
                "embeddings": embeddings,
            }
        return None


# %%
def main(
    db_path: str = constants.DEFAULT_DB, text_dir: str = constants.TEXT_DIRECTORY
) -> None:
    """
    Process all character documents and build a knowledge graph.

    Executes the full pipeline:
    1. Initialize database
    2. Read character documents from directory
    3. Process each document (split, embed)
    4. Extract relationships
    5. Save all data to the database

    Args:
        db_path: Path to SQLite database file
        text_dir: Directory containing character text files

    Returns:
        None

    """
    logger.info(f"Initializing database at {db_path}")
    init_database(db_path)

    # Get all text files from directory
    text_path = Path(text_dir)
    if not text_path.exists() or not text_path.is_dir():
        logger.error(f"Text directory not found: {text_dir}")
        return

    character_files = list(text_path.glob("*.txt"))
    logger.info(f"Found {len(character_files)} character files")

    all_characters: list[Character] = []
    all_relationships: list[Relationship] = []

    # Process each character file
    for file_path in character_files:
        char_id = file_path.stem  # Use filename without extension as character ID
        logger.info(f"Processing character: {char_id}")

        try:
            # Read document
            content = read_file(file_path)

            # Split into chunks
            chunks = split_text(
                content,
                chunk_size=constants.CHUNK_SIZE,
                overlap=constants.CHUNK_OVERLAP,
                separators=constants.SEPARATORS,
            )

            if not chunks:
                logger.warning(f"No chunks generated for {char_id}, skipping")
                continue

            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            embeddings = np.array(embed_text_chunks(chunks), dtype=np.float32)

            # Save document with chunks and embeddings
            logger.info(f"Saving document data for {char_id}")
            save_document(db_path, char_id, content, chunks, embeddings)

            # Extract relationships
            logger.info(f"Extracting relationships from {char_id}")
            parsed_results = extract_relationships(content)

            # Process extracted relationships
            for line_results in parsed_results.values():
                if len(line_results) == 3:  # Valid relationship entry
                    char1_data, char2_data, rel_data = line_results

                    char1 = Character(**char1_data)
                    char2 = Character(**char2_data)
                    rel = Relationship(**rel_data)

                    # Add characters and relationship to collections
                    all_characters.extend([char1, char2])
                    all_relationships.append(rel)

        except Exception as e:
            logger.error(f"Error processing {char_id}: {e}")

    # Save all graph data
    if all_characters and all_relationships:
        logger.info(
            f"Saving graph data: {len(all_characters)} characters, {len(all_relationships)} relationships"
        )
        save_graph_data(db_path, all_characters, all_relationships)
    else:
        logger.warning("No relationships or characters extracted")

    logger.info("Processing complete")


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process character documents and build knowledge graph"
    )
    parser.add_argument(
        "--db", default=constants.DEFAULT_DB, help="Path to SQLite database"
    )
    parser.add_argument(
        "--texts",
        default=constants.TEXT_DIRECTORY,
        help="Directory containing character text files",
    )

    args = parser.parse_args()
    main(args.db, args.texts)
