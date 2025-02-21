"""This module defines a set of constants for use in a text processing application,
including configuration options for logging, chunking, and client settings."""

from logging import Logger

from ollama import Client
from pydantic import BaseModel as PydanticBaseModel

from ppcs.logger import setup_logger

# %% [markdown]
# ## Constants


# %%
class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


# %%
class Constants(BaseModel):
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = int(0.2 * CHUNK_SIZE)
    EXAMPLE_TEXT_DIRECTORY: str = "example_text"
    SEPARATORS: list[str] = ["\n\n", "\n", ".", " ", ""]
    MODEL: str = "phi4:latest"
    CLIENT: Client = Client(host="http://localhost:11434")
    LOG_LEVEL: str = "INFO"
    LOGGER: Logger = setup_logger(LOG_LEVEL)


# %%
# Constants instance
constants = Constants(
    CHUNK_SIZE=1000,
    CHUNK_OVERLAP=200,
    EXAMPLE_TEXT_DIRECTORY="example_text",
    SEPARATORS=["\n\n", "\n", ".", " ", ""],
    MODEL="phi4:latest",
    CLIENT=Client(host="http://localhost:11434"),
    LOG_LEVEL="INFO",
    LOGGER=setup_logger(),
)
