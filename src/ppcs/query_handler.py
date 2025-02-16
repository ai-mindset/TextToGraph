"""This module handles queries by calculating centrality measures, summarizing them,
and using a language model to generate a response based on both the query
and the centrality summary."""

from ollama import Client

from ppcs.graph_manager import GraphManager
from ppcs.logger import Logger


# %%
class QueryHandler:
    logger = Logger("QueryHandler").get_logger()

    def __init__(self, graph_manager: GraphManager, client: Client, model: str) -> None:
        self.graph_manager = graph_manager
        self.client = client
        self.model = model

    def ask_question(self, query: str) -> str:
        centrality_data = self.graph_manager.calculate_centrality_measures()
        centrality_summary = self.graph_manager.summarize_centrality_measures(
            centrality_data
        )

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """Use the centrality measures to answer 
                    the following query.""",
                },
                {
                    "role": "user",
                    "content": f"Query: {query} Centrality Summary: {centrality_summary}",
                },
            ],
        )
        self.logger.debug("Query answered: %s", response["message"]["content"])
        final_answer = response["message"]["content"]

        return final_answer
