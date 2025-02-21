"""  """


# %%
import re
import pickle
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from ppcs.constants import constants

# %%
logger = constants.LOGGER

# %% ✅
class Character(BaseModel):
    """Character model with traits and relationships."""
    id: str
    traits: list[str] = Field(default_factory=list)


class Relationship(BaseModel):
    """Relationship between two characters."""
    source: str
    target: str
    relationship: str
    weight: float = Field(ge=0.0, le=1.0)


# %% ✅
@contextmanager
def get_db_connection(db_path: str) -> Generator:
    """Create and manage database connection with proper error handling.

    Args:
        db_path: Path to SQLite database

    Yields:
        SQLite connection object

    Examples:
        >>> with get_db_connection(":memory:") as conn:
        ...     isinstance(conn, sqlite3.Connection)
        True

    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    finally:
        if conn is not None:
            conn.close()


# %% ✅
def read_document(file_path: Path) -> str:
    """Read document content from a file with error handling.

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
        ...     content = read_document(Path(f.name))
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
        >>> splitter = RecursiveCharacterTextSplitter(
        ... chunk_size=100,
        ... chunk_overlap=20,
        ... separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        ... )
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
def embed_text_batch(texts: list[str]) -> list[list[float]]:
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
def parse_relationship_line(line: str) -> tuple[Character | None, Relationship | None]:
    """Parse a single relationship line from LLM output.

    Args:
        line: Single line of LLM output

    Returns:
        Tuple of (Character | None, Relationship | None)

    Examples:
        >>> line = "Character1 (Alice) -> Related to -> Character2 (Bob) [strength: 1.0]"
        >>> parse_relationship_line(line)[1]
        Relationship(source='Alice', target='Bob', relationship='Related to', weight=1.0)
        >>> parse_relationship_line("Character1 traits: brave, strong")[0]
        Character(id='Character1', traits=['brave', 'strong'])
        >>> parse_relationship_line("Character1 traits:")[0]
        Character(id='Character1', traits=[])
        >>> parse_relationship_line("Character1")[0]
        Character(id='Character1', traits=[])
        >>> parse_relationship_line("    ") == (None, None)
        True
    """
    # Strip whitespace to normalize input
    line = line.strip()
    if not line:
        return None, None

    # Try to match relationship pattern first
    if "->" in line and "[strength:" in line:
        # Extract names in parentheses
        names = re.findall(r'\(([^)]+)\)', line)
        if len(names) != 2:
            return None, None
        
        source, target = [name.strip() for name in names]
        
        # Extract relationship
        rel_parts = line.split("->")
        if len(rel_parts) != 3:
            return None, None
            
        relationship = rel_parts[1].strip()
        
        # Extract weight
        weight_match = re.search(r'\[strength:\s*(\d*\.?\d+)\]', line)
        if not weight_match:
            return None, None
            
        try:
            weight = float(weight_match.group(1))
            if not 0.0 <= weight <= 1.0:
                return None, None
                
            return None, Relationship(
                source=source,
                target=target,
                relationship=relationship,
                weight=weight
            )
        except ValueError:
            return None, None

    # Try to match traits pattern
    traits_match = re.match(r'^(\w+)(?:\s+traits:(?:\s*(.+))?)?$', line)
    if traits_match:
        char_id = traits_match.group(1)
        traits_str = traits_match.group(2)
        
        traits = []
        if traits_str:
            traits_str = traits_str.strip()
            if traits_str.upper() != 'N/A':
                traits = [t.strip() for t in traits_str.split(',')]
                
        return Character(id=char_id, traits=traits), None

    return None, None


# %% ✅ 
def is_valid_relationship_line(line: str) -> bool:
    """Check if a line contains a valid relationship format.
    
    Args:
        line: Line to validate
    
    Returns:
        bool: True if line matches expected format
        
    Examples:
        >>> is_valid_relationship_line("Parsed relationship: Alice traits: brave. Alice -> friend -> Bob [strength: 0.8]")
        True
        >>> is_valid_relationship_line("Random text")
        False
    """
    # Normalize line for comparison
    normalised = line.lower().strip()
    
    # Check structural elements that must be present
    required_elements = [
        # "traits:",  # Must have traits section
        "->",       # Must have relationship arrow
        "strength", # Must have strength indicator
        "["        # Must have strength bracket
    ]
    
    return any(element in normalised for element in required_elements)


%% ✅
def extract_relationships(text: str) -> tuple[list[Character], list[Relationship]]:
    """Extract relationships from text using Mistral.

    Args:
        text: Text to analyse

    Returns:
        Tuple of (list[Character], list[Relationship])

    Examples:
        >>> text = "Alice is Bob's sister. They are very close."
        >>> chars, rels = extract_relationships(text)
        >>> len(chars) > 0 and len(rels) > 0
        True

    """
    if not text.strip():
        return [], []

    try:
        llm = OllamaLLM(model="mistral", temperature=0.0)
        system_prompt = """Extract characters, character traits, relationships with other characters and each relationship strength from the following text. Use common terms such as 'related to', 'depends on', 'influences', etc., for relationships. Traits could be 'strength', 'stamina', 'resourceful' etc. Estimate a relationship's strength between 0.0 (very weak) and 1.0 (very strong). Format: Parsed relationship: Character1 -> Relationship -> Character2 [strength: X.X]. Do not include any other text in your response. Use this exact format:   
        Parsed relationship: 
        Character1 traits: trait1, trait2, ... . 
        Character1 -> Relationship -> Character2 [strength: X.X]."""

        response = llm.invoke(f"{system_prompt}\n\nText: {text}")

        characters = []
        relationships = []

        parsed_results = []
        parsing_errors = []
        
        for i, line in enumerate(response.split("\n"), 1):
            line = line.strip().lower()
            is_valid = is_valid_relationship_line(line)
            if not line or not is_valid:
                parsing_errors.append(f"Line {i}: Skipped invalid line format: '{line}'")
                continue
                
            try:
                logger.info(f"Parsing: {line}\b")
                character, relationship = parse_relationship_line(line)
                parsed_results.append((character, relationship))
            except ValueError as e:
                parsing_errors.append(f"Line {i}: {str(e)}")

        if parsing_errors:
            raise ValueError(
                "Some relationships could not be parsed:\n" +
                "\n".join(parsing_errors)
            )
        
        if not parsed_results:
            raise ValueError("No valid relationships found in text")
            
        # Unzip the results
        characters, relationships = zip(*parsed_results) if parsed_results else ([], [])

        return list(characters), list(relationships)
    except Exception as e:
        raise RuntimeError(f"Relationship extraction failed: {e}")




def init_database(db_path: str) -> None:
    """Initialize SQLite database with required tables and indices.

    Args:
        db_path: Path to SQLite database
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


def save_document(
    db_path: str,
    doc_id: str,
    content: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> None:
    """Save document and its processed data to database.

    Args:
        db_path: Path to SQLite database
        doc_id: Document identifier
        content: Original document content
        chunks: List of text chunks
        embeddings: List of embeddings for each chunk
    """
    if not doc_id or not content:
        raise ValueError("Document ID and content are required")

    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks and embeddings must match")

    with get_db_connection(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents (id, content, chunks, embeddings) VALUES (?, ?, ?, ?)",
            (
                doc_id,
                content,
                pickle.dumps(chunks),
                pickle.dumps(np.array(embeddings, dtype=np.float32)),
            ),
        )


def save_graph_data(
    db_path: str, characters: list[Character], relationships: list[Relationship]
) -> None:
    """Save character and relationship data to database.

    Args:
        db_path: Path to SQLite database
        characters: List of characters with their traits
        relationships: List of relationships between characters
    """
    if not characters and not relationships:
        return

    with get_db_connection(db_path) as conn:
        # Save unique characters
        unique_chars = {char.id: char for char in characters}

        for character in unique_chars.values():
            conn.execute(
                "INSERT OR REPLACE INTO nodes (id, properties) VALUES (?, ?)",
                (character.id, json.dumps(asdict(character))),
            )

        # Save relationships
        for rel in relationships:
            conn.execute(
                "INSERT OR REPLACE INTO edges (source, target, relationship, weight) VALUES (?, ?, ?, ?)",
                (rel.source, rel.target, rel.relationship, rel.weight),
            )
