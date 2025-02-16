"""This module provides a class `GraphDatabaseConnection` that manages a SQLite database
for storing and querying graph data, including nodes and edges with optional properties
and relationships"""

import sqlite3
from sqlite3.dbapi2 import Connection


# %%
class GraphDatabaseConnection:
    def __init__(self, db_path: str = "data/graph_database.sqlite") -> None:
        if not db_path:
            raise ValueError(
                "Database path must be provided to initialize the DatabaseConnection."
            )
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        # Use WAL mode for better performance
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.initialize_schema()

    def initialize_schema(self) -> None:
        with self.conn:
            # Create nodes and edges tables if they don't exist
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    properties TEXT
                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    source TEXT,
                    target TEXT,
                    relationship TEXT,
                    weight REAL,
                    PRIMARY KEY (source, target, relationship),
                    FOREIGN KEY (source) REFERENCES nodes(id),
                    FOREIGN KEY (target) REFERENCES nodes(id)
                )
            """)

            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS source_idx ON edges(source)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS target_idx ON edges(target)
            """)

    def close(self) -> None:
        self.conn.close()

    def get_session(self) -> Connection:
        return self.conn

    def clear_database(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM edges")
            self.conn.execute("DELETE FROM nodes")
