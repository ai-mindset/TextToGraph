"""This module defines a set of constants for use in a text processing application,
including configuration options for logging, chunking, and client settings."""

from ollama import Client
from pydantic import BaseModel as PydanticBaseModel

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
    TEXT_DIRECTORY: str = "text"
    SEPARATORS: list[str] = ["\n\n", "\n", ".", " ", ""]
    MODEL: str = "phi4:latest"
    CLIENT: Client = Client(host="http://localhost:11434")
    LOG_LEVEL: str = "INFO"
