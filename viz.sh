#!/usr/bin/env zsh

print -P "%F{green}Export graph data from a SQLite database to a JSON file in a format suitable for use with D3.js%f"
python src/ppcs/export_graph_data.py data/graph_database.sqlite

print -P "%F{green}Visualise graph using D3.js%f"
python -m http.server --directory public 8000
