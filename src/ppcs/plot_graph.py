import json
import os
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from ppcs.constants import Constants
from ppcs.logger import setup_logger

# Initialize constants
constants = Constants()
logger = setup_logger(constants.LOG_LEVEL)


def load_and_visualize_graph(
    db_path: str | None = None,
    output_path: str | None = None,
    layout_algo: str = "fruchterman_reingold",
    min_weight: float = 0.0,
    color_map: dict[str, str] | None = None,
    height: int = 800,
    width: int = 1200,
) -> go.Figure:
    """
    Load relationship graph from database and create an interactive visualization.

    Args:
        db_path: Path to the SQLite database
        output_path: Path to save HTML output (optional)
        layout_algo: Graph layout algorithm ('fruchterman_reingold', 'kamada_kawai', or 'spring')
        min_weight: Minimum relationship weight to include (0.0-1.0)
        color_map: Dictionary mapping relationship types to colors
        height: Height of the plot in pixels
        width: Width of the plot in pixels

    Returns:
        Plotly figure object containing the interactive graph

    Examples:
        >>> # Basic usage
        >>> fig = load_and_visualize_graph("world.db")
        >>> # Save output and filter by weight
        >>> fig = load_and_visualize_graph("world.db", "characters_graph.html", min_weight=0.3)
        >>> # Custom colors for relationship types
        >>> colors = {"friend": "green", "enemy": "red", "family": "blue"}
        >>> fig = load_and_visualize_graph("world.db", color_map=colors)
    """
    # Default color mapping if none provided
    if color_map is None:
        color_map = {
            "friend": "#4CAF50",  # Green
            "enemy": "#F44336",  # Red
            "family": "#2196F3",  # Blue
            "colleague": "#FF9800",  # Orange
            "spouse": "#E91E63",  # Pink
            "sibling": "#9C27B0",  # Purple
            "influences": "#00BCD4",  # Cyan
            "depends on": "#795548",  # Brown
            "related to": "#607D8B",  # Blue Grey
        }

    # Default color for relationships not in the mapping
    default_color = "#9E9E9E"  # Grey

    # Use default database path from constants if not provided
    if db_path is None:
        db_path = constants.DEFAULT_DB

    # Ensure the database directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Connect to database and load graph data
    logger.info(f"Loading graph data from {db_path}")

    try:
        from ppcs.main import get_db_connection

        # Create a directed graph to represent relationships
        G = nx.DiGraph()

        # Load nodes and edges from database
        with get_db_connection(db_path) as conn:
            # Load nodes
            cursor = conn.execute("SELECT id, properties FROM nodes")
            nodes = cursor.fetchall()

            for node_id, properties_json in nodes:
                properties = json.loads(properties_json)
                G.add_node(node_id, **properties)

            # Load edges with weight >= min_weight
            cursor = conn.execute(
                "SELECT source, target, relationship, weight FROM edges WHERE weight >= ?",
                (min_weight,),
            )
            edges = cursor.fetchall()

            for source, target, relationship, weight in edges:
                G.add_edge(source, target, relationship=relationship, weight=weight)

        if not G.nodes:
            logger.warning("No nodes found in the database")
            return go.Figure()

        if not G.edges:
            logger.warning(f"No relationships found with weight >= {min_weight}")
            # Still show nodes even if no edges meet criteria

        # Calculate node positions based on selected layout algorithm
        pos = _calculate_layout(G, layout_algo)

        # Create the figure
        fig = _create_plotly_graph(G, pos, color_map, default_color, height, width)

        # Save to HTML if output path is provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(output_file)
            logger.info(f"Graph visualization saved to {output_path}")

        return fig

    except Exception as e:
        logger.error(f"Error visualizing graph: {e}")
        raise


def _calculate_layout(G: nx.DiGraph, algorithm: str) -> dict[str, tuple[float, float]]:
    """
    Calculate node positions using the specified layout algorithm.

    Args:
        G: NetworkX graph
        algorithm: Layout algorithm name

    Returns:
        Dictionary mapping node IDs to (x, y) positions
    """
    logger.info(f"Calculating graph layout using {algorithm} algorithm")

    # Set random seed for reproducibility
    seed = 42

    # Add some randomness to initial positions to avoid overlaps
    initial_pos = {node: (np.random.random(), np.random.random()) for node in G.nodes()}

    if algorithm == "fruchterman_reingold":
        return nx.fruchterman_reingold_layout(
            G, k=0.3, iterations=100, seed=seed, pos=initial_pos
        )
    elif algorithm == "kamada_kawai":
        return nx.kamada_kawai_layout(G, pos=initial_pos)
    elif algorithm == "spring":
        return nx.spring_layout(G, k=0.5, iterations=100, seed=seed, pos=initial_pos)
    else:
        logger.warning(
            f"Unknown layout algorithm '{algorithm}', using fruchterman_reingold"
        )
        return nx.fruchterman_reingold_layout(
            G, k=0.3, iterations=100, seed=seed, pos=initial_pos
        )


def _create_plotly_graph(
    G: nx.DiGraph,
    pos: dict[str, tuple[float, float]],
    color_map: dict[str, str],
    default_color: str,
    height: int,
    width: int,
) -> go.Figure:
    """
    Create a Plotly figure from the NetworkX graph.

    Args:
        G: NetworkX graph
        pos: Dictionary mapping node IDs to positions
        color_map: Dictionary mapping relationship types to colors
        default_color: Default color for undefined relationships
        height: Height of the plot in pixels
        width: Width of the plot in pixels

    Returns:
        Plotly figure
    """
    # Extract node positions
    x_pos = [pos[node][0] for node in G.nodes()]
    y_pos = [pos[node][1] for node in G.nodes()]

    # Create a placeholder for the figure
    fig = go.Figure()

    # Add edges (drawn as annotations to support arrows)
    edge_traces = []

    # Group edges by relationship type for legend
    relationship_edges = {}

    for source, target, edge_data in G.edges(data=True):
        x0, y0 = pos[source]
        x1, y1 = pos[target]

        relationship = edge_data.get("relationship", "unknown")
        weight = edge_data.get("weight", 0.5)

        # Create hover text with more details
        hover_text = (
            f"<b>{source}</b> {relationship} <b>{target}</b><br>Strength: {weight:.2f}"
        )

        # Get color for this relationship type
        color = color_map.get(relationship.lower(), default_color)

        # Scale line_width by weight (0.5 - 3.0)
        line_width = 0.5 + 2.5 * weight

        # Add to relationship group
        if relationship not in relationship_edges:
            relationship_edges[relationship] = []

        relationship_edges[relationship].append((x0, y0, x1, y1, line_width, hover_text))

    # Add one trace per relationship type (for legend)
    for relationship, edges in relationship_edges.items():
        color = color_map.get(relationship.lower(), default_color)

        # Add representative edge for the legend
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=relationship,
                line=dict(
                    color=color,
                    width=2,
                ),
                hoverinfo="none",
                legendgroup=relationship,
            )
        )

        # Add all edges of this relationship type
        for x0, y0, x1, y1, line_width, hover_text in edges:
            # Add edge line
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(color=color, width=line_width),
                    hoverinfo="text",
                    hovertext=hover_text,
                    legendgroup=relationship,
                    showlegend=False,
                )
            )

            # Add arrow at target
            angle = np.arctan2(y1 - y0, x1 - x0)
            dx = 0.03 * np.cos(angle)
            dy = 0.03 * np.sin(angle)

            fig.add_annotation(
                x=x1,
                y=y1,
                ax=x1 - dx,
                ay=y1 - dy,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=line_width,
                arrowcolor=color,
                opacity=0.8,
            )

    # Add nodes with better styling
    node_trace = go.Scatter(
        x=x_pos,
        y=y_pos,
        mode="markers+text",
        marker=dict(
            size=20,
            color="#FFFFFF",  # White fill
            line=dict(width=2, color="#444444"),  # Dark outline
            opacity=0.9,
            symbol="circle",
        ),
        text=[node_id for node_id in G.nodes()],
        textposition="top center",
        textfont=dict(size=14, color="#000000"),
        hoverinfo="text",
        hovertext=[_format_node_hover(G, node_id) for node_id in G.nodes()],
    )

    fig.add_trace(node_trace)

    # Update layout for better visualization
    fig.update_layout(
        title=dict(
            text="Character Relationship Graph", x=0.5, font=dict(family="Arial", size=16)
        ),
        font=dict(family="Arial", size=12),
        showlegend=True,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="Hover over nodes and edges for details. Use toolbar to zoom, pan, or reset view.",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.01,
                y=-0.05,
            )
        ],
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]
        ),
        plot_bgcolor="rgba(240,240,240,0.8)",
        height=height,
        width=width,
        legend=dict(
            title="Relationship Types",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            itemsizing="constant",
        ),
    )

    # Add zoom and pan capabilities
    fig.update_layout(dragmode="pan", modebar=dict(remove=["select", "lasso"]))

    return fig


def _format_node_hover(G: nx.DiGraph, node_id: str) -> str:
    """
    Format hover text for a node showing its properties.

    Args:
        G: NetworkX graph
        node_id: ID of the node

    Returns:
        Formatted hover text
    """
    node_data = G.nodes[node_id]

    # Count incoming and outgoing relationships
    in_edges = list(G.in_edges(node_id, data=True))
    out_edges = list(G.out_edges(node_id, data=True))

    in_count = len(in_edges)
    out_count = len(out_edges)

    hover_parts = [f"<b>{node_id}</b>"]

    # Add traits if available
    try:
        if isinstance(node_data, dict) and "traits" in node_data:
            traits = node_data["traits"]
            if isinstance(traits, list) and traits:
                trait_text = ", ".join(traits)
                hover_parts.append(f"Traits: {trait_text}")
    except Exception:
        pass

    # Add relationship counts
    hover_parts.append(f"Incoming relationships: {in_count}")
    hover_parts.append(f"Outgoing relationships: {out_count}")

    return "<br>".join(hover_parts)


if __name__ == "__main__":
    import argparse

    # Get default database path from constants
    default_db = constants.DEFAULT_DB

    parser = argparse.ArgumentParser(description="Visualize character relationship graph")
    parser.add_argument(
        "--db",
        default=default_db,
        help=f"Path to SQLite database (default: {default_db})",
    )
    parser.add_argument(
        "--output", default="character_graph.html", help="Path to save HTML output"
    )
    parser.add_argument(
        "--layout",
        default="fruchterman_reingold",
        choices=["fruchterman_reingold", "kamada_kawai", "spring"],
        help="Layout algorithm to use",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.0,
        help="Minimum relationship weight to include (0.0-1.0)",
    )
    parser.add_argument(
        "--height", type=int, default=800, help="Height of the plot in pixels"
    )
    parser.add_argument(
        "--width", type=int, default=1200, help="Width of the plot in pixels"
    )

    args = parser.parse_args()

    # Generate and save the visualization
    load_and_visualize_graph(
        db_path=args.db,
        output_path=args.output,
        layout_algo=args.layout,
        min_weight=args.min_weight,
        height=args.height,
        width=args.width,
    )

    print(f"Graph visualization saved to {args.output}")
