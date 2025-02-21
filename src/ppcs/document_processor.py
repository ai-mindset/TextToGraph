"""Process documents. Extract relationship strength in the following format:
Entity1 -> Relationship -> Entity2 [strength: X.X] (strength [0.0 1.0]"""

import os
import pickle

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ppcs.constants import constants

model = constants.MODEL
logger = constants.LOGGER
client = constants.CLIENT


# %%
def load_documents(directory: str = constants.EXAMPLE_TEXT_DIRECTORY) -> list[str]:
    """
    Loads text documents from the specified directory.

    Args:
        directory (str): The path to the directory containing document files.
        Defaults to `constants.EXAMPLE_TEXT_DIRECTORY`.

    Returns:
        list[str]: A list of strings, where each string is a content of a document file
        that starts with "doc_" and ends with ".txt".

    Examples:
        >>> from ppcs.constants import constants
        >>> load_documents(constants.EXAMPLE_TEXT_DIRECTORY)
        ['doc_1.txt', 'doc_4.txt']

    """
    documents = []
    for file in os.listdir(directory):
        if file.startswith("doc_") and file.endswith(".txt"):
            with open(os.path.join(directory, file)) as f:
                documents.append(f.read())

    return documents


# %%
def split_documents(
    documents: list[str],
    chunk_size: int = constants.CHUNK_SIZE,
    chunk_overlap: int = constants.CHUNK_OVERLAP,
    separators: list[str] = constants.SEPARATORS,
) -> list[str]:
    """
    Split documents into semantic chunks using recursive character splitting.

    Args:
        documents: list of document strings to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        separators: Tuple of separators to use, in order of preference

    Returns:
        list of semantically split chunks

    Raises:
        ValueError: If chunk_size <= chunk_overlap
    """
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = []
    for doc in documents:
        if not isinstance(doc, str):
            raise TypeError(f"Expected string, got {type(doc)}")
        chunks.extend(splitter.split_text(doc))

    return chunks


# %%
def extract_elements(chunks: list[str]) -> list[str]:
    elements = []
    for index, chunk in enumerate(chunks):
        logger.debug(
            f"Extracting elements and relationship strength from chunk {index + 1}"
        )
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": """Extract entities, relationships, and their strength 
                    from the following text. Use common terms such as 'related to', 
                    'depends on', 'influences', etc., for relationships, and estimate 
                    a strength between 0.0 (very weak) and 1.0 (very strong). 
                    Format: Parsed relationship: 
                    Entity1 -> Relationship -> Entity2 [strength: X.X]. 
                    Do not include any other text in your response. 
                    Use this exact format: Parsed relationship: 
                    Entity1 -> Relationship -> Entity2 [strength: X.X].""",
                },
                {"role": "user", "content": chunk},
            ],
        )
        entities_and_relations = response["message"]["content"]
        elements.append(entities_and_relations)
    logger.debug("Elements extracted")

    return elements


# %%
def summarize_elements(elements: list[str]) -> list[str]:
    summaries = []
    for index, element in enumerate(elements):
        logger.debug(f"Summarizing element {index + 1}")
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": """Summarize the following entities and relationships 
                    in a structured format. Use common terms such as 'related to', 
                    'depends on', 'influences', etc., for relationships. Use '->' 
                    to represent relationships after the 'Relationships:' word.""",
                },
                {"role": "user", "content": element},
            ],
        )
        summary = response["message"]["content"]
        summaries.append(summary)
    logger.debug("Summaries created")

    return summaries


# %%
# ## Main guard

if __name__ == "__main__":
    docs = load_documents(directory=constants.EXAMPLE_TEXT_DIRECTORY)

    # Split documents into chunk_size
    chunks = split_documents(documents=docs)

    # Extract elements and relationship strength
    elements = extract_elements(chunks=chunks)

    # Summarize Elements
    summaries = summarize_elements(elements=elements)
