# TextToGraph

A Python-based system that transforms text files into interactive relationship graphs by extracting character relationships using LLMs, storing data in SQLite, and generating network visualisations.

## Overview

TextToGraph analyses text documents containing character information, extracts relationships between characters using Large Language Models (LLMs), and creates a knowledge graph database of characters and their relationships. It then visualises these relationships as an animated network graph.

## Features

- Text processing with chunk-based analysis
- Character and relationship extraction using Ollama LLMs
- Persistent storage using SQLite
- Advanced network visualisation with:
  - Colourblind-friendly design
  - Interactive animation
  - Optimised node layout
  - Relationship strength indicators

## Requirements

- Python 3.10+
- Ollama server running locally (with mistral-small and nomic-embed-text models)
- Required Python packages:
  - langchain
  - langchain_ollama
  - langchain_text_splitters
  - matplotlib
  - networkx
  - numpy
  - pydantic

## Project Structure

- `constants.py`: Configuration parameters for the application
- `logger.py`: Logging utilities
- `main.py`: Core functionality for processing texts and building the knowledge graph
- `plot_graph.py`: Network visualisation tools
- `__init__.py`: Package initialisation

## Usage

### 1. Set up the directory structure

```
├── data/           # Database storage
├── plots/          # Generated visualisations
├── text/           # Input text files (.txt or .md)
└── texttograph/    # The code module
```

### 2. Prepare your text files

Place character description text files in the `text/` directory. Each file should contain information about a character and their relationships with other characters. The filename (without extension) will be used as the character ID.

### 3. Start the Ollama server

Ensure the Ollama server is running locally with the required models:

```bash
ollama serve
```

### 4. Run the main processing script

```bash
python -m texttograph.main
```

This will:
- Read all text files from the `text/` directory
- Process and chunk the content
- Generate embeddings
- Extract character relationships using LLMs
- Store all data in the SQLite database

### 5. Generate the visualisation

```bash
python -m texttograph.plot_graph
```

This will create an animated network graph of character relationships and save it to the path specified in `constants.py` (default: `plots/character_graph.mp4`).

## Configuration

Edit `constants.py` to configure:

- `CHUNK_SIZE`: Size of text chunks for processing (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 20% of chunk size)
- `PLOT_DIRECTORY`: Where visualisations are saved (default: "plots")
- `TEXT_DIRECTORY`: Where input text files are located (default: "text")
- `DB_DIRECTORY`: Database storage location (default: "data")
- `MODEL`: Ollama model to use (default: "mistral-small:24b-instruct-2501-q4_K_M")
- `LOG_LEVEL`: Logging verbosity (default: "INFO")

## How It Works

1. **Text Processing**: 
   - Reads text files from the `text/` directory
   - Splits content into overlapping chunks
   - Generates embeddings for each chunk

2. **Relationship Extraction**:
   - Uses LLMs to identify characters, traits, and relationships in the text
   - Parses the output into structured `Character` and `Relationship` objects

3. **Database Storage**:
   - Initialises an SQLite database with tables for documents, nodes, and edges
   - Stores the original content, chunks, embeddings, characters, and relationships

4. **Network Visualisation**:
   - Creates a directed graph using NetworkX
   - Assigns visually distinct attributes based on character grouping
   - Generates an animation showing the network formation

## Advanced Usage

### Custom Database Path

```bash
python -m texttograph.main --db path/to/database.db
```

### Custom Text Directory

```bash
python -m texttograph.main --texts path/to/texts
```
