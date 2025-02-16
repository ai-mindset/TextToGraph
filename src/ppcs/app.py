"""Main app"""

# %% [markdown]
# ## Imports

# %%
import os
import pickle
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from ollama import Client

from ppcs.constants import DOCUMENTS, DOCUMENTS_TO_ADD_TO_INDEX
from ppcs.document_processor import DocumentProcessor
from ppcs.graph_database import GraphDatabaseConnection
from ppcs.graph_manager import GraphManager
from ppcs.logger import Logger
from ppcs.query_handler import QueryHandler

# %% [markdown]
# ## Load env vars

# %%
load_dotenv()

DB_PATH = os.getenv("DB_PATH")
if not DB_PATH:
    raise ValueError("No value for 'DB_PATH' found in .env")

# ## [markdown]
# ## Initialise

# Client initialization
client = Client(host="http://localhost:11434")
MODEL = "phi4:latest"

# Initialize document processor
doc_processor = DocumentProcessor(client, MODEL)

# Initialize database connection
db_connection = GraphDatabaseConnection(db_path=DB_PATH)

# Initialize graph manager
graph_manager = GraphManager(db_connection)

# Initialize query handler
query_handler = QueryHandler(graph_manager, client, MODEL)

# Initialize logger
logger = Logger("PPCs Logger").get_logger()

# %% [markdown]
# ##Functions related to document processing


# %%
def load_or_run(file_path: str, run_function: Callable, *args: list) -> Any:
    """Loads data from a file if it exists, otherwise runs a function to generate
    and save the data.

    Args:
        file_path (str): The path to the file where the data is stored or will be saved.
        run_function (callable): A function that generates data when called with `*args`.
        *args (list): Arguments passed to the `run_function`.

    Returns:
        any: The loaded or generated data.

    Examples:
        >>> with open("data/data.pkl", "wb") as f:
        ...    import pickle; pickle.dump({"key": 1}, f)
        >>> def get_data():
        ...     return {"key": "value"}
        >>> data = load_or_run("data/data.pkl", get_data)
        Loading function to generate data for data.pkl
        {'key': 'value'}
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory {directory}")

    if os.path.exists(file_path):
        logger.info(f"Loading data from {file_path}")
        with open(file_path, "rb") as file:
            data = pickle.load(file)
    else:
        logger.info(f"Running function to generate data for {file_path}")
        data = run_function(*args)
        if data is not None:
            with open(file_path, "wb") as file:
                pickle.dump(data, file)

    return data


# %%
def initial_indexing(documents: list[str], graph_manager: GraphManager) -> None:
    """Indexes documents by processing them into chunks,
    extracting elements and summaries, and building a graph.

    Args:
        documents (list[str]): List of document strings.
        graph_manager (GraphManager): Instance of the GraphManager, to build the graph.

    Returns:
        None
    """
    chunks = doc_processor.split_documents(documents)
    elements_file = "data/initial_elements_data.pkl"
    summaries_file = "data/initial_summaries_data.pkl"

    elements = load_or_run(elements_file, doc_processor.extract_elements, chunks)
    summaries = load_or_run(summaries_file, doc_processor.summarize_elements, elements)

    graph_manager.build_graph(summaries)


# %%
def reindex_with_new_documents(
    new_documents: list[str], graph_manager: GraphManager
) -> None:
    """Reindexes the document database using new documents.

    Args:
        new_documents (list[str]): A list of new document contents.
        graph_manager (GraphManager): The graph manager to use for building and
        reprojecting the graph.

    Returns:
        None
    """
    chunks = doc_processor.split_documents(new_documents)
    elements_file = "data/new_elements_data.pkl"
    summaries_file = "data/new_summaries_data.pkl"

    elements = load_or_run(elements_file, doc_processor.extract_elements, chunks)
    summaries = load_or_run(summaries_file, doc_processor.summarize_elements, elements)

    graph_manager.build_graph(summaries)
    graph_manager.reproject_graph()


# %%
if __name__ == "__main__":
    initial_documents = DOCUMENTS

    # Index the initial documents
    initial_indexing(initial_documents, graph_manager)

    # First question after initial indexing
    query_1 = "What are the main themes in these documents?"
    logger.info("Query 1: %s", query_1)
    answer_1 = query_handler.ask_question(query_1)
    logger.info("Answer to query 1: %s", answer_1)

    # Adding new documents and reindexing
    new_documents = DOCUMENTS_TO_ADD_TO_INDEX
    reindex_with_new_documents(new_documents, graph_manager)

    # Second question after reindexing with new documents
    query_2 = "What are the main themes in these documents?"
    logger.info("Query 2: %s", query_2)
    answer_2 = query_handler.ask_question(query_2)
    logger.info("Answer to query 2: %s", answer_2)
