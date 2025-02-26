"""This module defines a set of constants for use in a text processing application,
including configuration options for logging, chunking, and client settings."""

from ollama import Client
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, computed_field

# %% [markdown]
# ## Constants


# %%
class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


class Constants(BaseModel):
    CHUNK_SIZE: int = Field(default=1000)

    @computed_field
    @property
    def CHUNK_OVERLAP(self) -> int:  # noqa: N802
        return int(0.2 * self.CHUNK_SIZE)

    PLOT_DIRECTORY: str = Field(default="plots")

    @computed_field
    @property
    def DEFAULT_PLOT(self) -> str:  # noqa: N802
        return self.PLOT_DIRECTORY + "/character_graph.html"

    TEXT_DIRECTORY: str = Field(default="text")
    DB_DIRECTORY: str = Field(default="data")

    @computed_field
    @property
    def DEFAULT_DB(self) -> str:  # noqa: N802
        return self.DB_DIRECTORY + "/world.db"

    SEPARATORS: list[str] = Field(default=["\n\n", "\n", ".", " ", ""])
    MODEL: str = Field(default="mistral-small:24b-instruct-2501-q4_K_M")
    CLIENT: Client = Field(default=Client(host="http://localhost:11434"))
    LOG_LEVEL: str = Field(default="INFO")
