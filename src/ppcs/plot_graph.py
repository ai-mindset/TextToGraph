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

# Add DEFAULT_PLOT to Constants class if needed
if not hasattr(constants, "DEFAULT_PLOT"):
    constants.DEFAULT_PLOT = os.path.join(constants.DB_DIRECTORY, "character_graph.html")


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
        db_path: Path to the SQLite database (defaults to constants.DEFAULT_DB)
        output_path: Path to save HTML output (defaults to constants.DEFAULT_PLOT)
        layout_algo: Graph layout algorithm ('fruchterman_reingold', 'kamada_kawai', or 'spring')
        min_weight: Minimum relationship weight to include (0.0-1.0)
        color_map: Dictionary mapping relationship types to colors
        height: Height of the plot in pixels
        width: Width of the plot in pixels

    Returns:
        Plotly figure object containing the interactive graph

    Examples:
        >>> # Basic usage with defaults
        >>> fig = load_and_visualize_graph()
        >>> # Save output and filter by weight
        >>> fig = load_and_visualize_graph(min_weight=0.3)
        >>> # Custom colors for relationship types
        >>> colors = {"friend": "green", "enemy": "red", "family": "blue"}
        >>> fig = load_and_visualize_graph(color_map=colors)
    """
    # Default color mapping if none provided - using a colorblind-friendly palette
    if color_map is None:
        color_map = {
            "friend": "#0072B2",  # Blue
            "enemy": "#D55E00",  # Vermilion (orange-red)
            "family": "#009E73",  # Green
            "colleague": "#CC79A7",  # Pink
            "spouse": "#56B4E9",  # Light blue
            "sibling": "#E69F00",  # Orange
            "influences": "#F0E442",  # Yellow
            "depends on": "#882255",  # Purple
            "related to": "#44AA99",  # Teal
        }

    # Default color for relationships not in the mapping
    default_color = "#9E9E9E"  # Grey

    # Use default database path and output path from constants if not provided
    if db_path is None:
        db_path = constants.DEFAULT_DB

    if output_path is None:
        output_path = constants.DEFAULT_PLOT

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
            # Load nodes (filtering out empty IDs)
            cursor = conn.execute("SELECT id, properties FROM nodes WHERE id != ''")
            nodes = cursor.fetchall()

            for node_id, properties_json in nodes:
                if not node_id:  # Skip nodes with empty IDs
                    continue
                properties = json.loads(properties_json)
                G.add_node(node_id, **properties)

            # Load edges with weight > 0.0 (filter out zero-weight relationships and empty source/target)
            cursor = conn.execute(
                """
                SELECT source, target, relationship, weight 
                FROM edges 
                WHERE weight > ? AND weight >= ? 
                AND source != '' AND target != ''
                """,
                (0.0, min_weight),
            )
            edges = cursor.fetchall()

            # Add edge only if both source and target exist in the graph
            for source, target, relationship, weight in edges:
                if source in G.nodes and target in G.nodes:
                    G.add_edge(source, target, relationship=relationship, weight=weight)

        # Check if graph has any nodes after filtering
        if not G.nodes:
            logger.warning("No valid nodes found in the database after filtering")
            fig = go.Figure()
            fig.update_layout(
                title="No Valid Nodes Found",
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
            )
            return fig

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

    # Get a fully connected version of the graph to improve layout
    if not nx.is_connected(G.to_undirected()):
        logger.info("Graph is not connected, adding weak connections for better layout")
        G_layout = G.copy()

        # Add weak connections between disconnected components to improve layout
        components = list(nx.connected_components(G.to_undirected()))
        if len(components) > 1:
            for i in range(len(components) - 1):
                src = next(iter(components[i]))
                tgt = next(iter(components[i + 1]))
                # Add a temporary edge with a very small weight
                G_layout.add_edge(src, tgt, weight=0.001, temp=True)
    else:
        G_layout = G

    # Calculate positions
    if algorithm == "fruchterman_reingold":
        # More iterations and higher k value spreads nodes better
        pos = nx.fruchterman_reingold_layout(G_layout, k=1.0, iterations=300, seed=seed)
    elif algorithm == "kamada_kawai":
        # Improve distance scaling for better spread
        pos = nx.kamada_kawai_layout(G_layout, scale=2.0)
    elif algorithm == "spring":
        # Higher k value and more iterations for better spread
        pos = nx.spring_layout(G_layout, k=1.5, iterations=200, seed=seed)
    else:
        logger.warning(
            f"Unknown layout algorithm '{algorithm}', using fruchterman_reingold"
        )
        pos = nx.fruchterman_reingold_layout(G_layout, k=1.0, iterations=300, seed=seed)

    return pos


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

        # Scale line_width by weight (1.0 - 5.0) - making weight differences more noticeable
        line_width = 1.0 + 4.0 * weight

        # Add to relationship group
        if relationship not in relationship_edges:
            relationship_edges[relationship] = []

        relationship_edges[relationship].append((x0, y0, x1, y1, line_width, hover_text))

    # Add one trace per relationship type (for legend)
    for relationship, edges in relationship_edges.items():
        color = color_map.get(relationship.lower(), default_color)

        # Show both color and pattern in legend for accessibility
        legend_name = f"{relationship} ({'—' * min(4, len(relationship))})"

        # Add representative edge for the legend
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=legend_name,
                line=dict(
                    color=color,
                    width=3,
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

    # Add nodes with accessibility-friendly styling
    node_trace = go.Scatter(
        x=x_pos,
        y=y_pos,
        mode="markers+text",
        marker=dict(
            size=20,
            color="#FFFFFF",  # White fill
            line=dict(width=2, color="#000000"),  # Black outline for high contrast
            opacity=1.0,  # Full opacity for better visibility
            symbol="circle",
        ),
        text=[node_id for node_id in G.nodes()],
        textposition="top center",
        textfont=dict(
            size=16, color="#000000", family="Arial"
        ),  # Larger, high-contrast text
        hoverinfo="text",
        hovertext=[_format_node_hover(G, node_id) for node_id in G.nodes()],
    )

    fig.add_trace(node_trace)

    # Update layout for better visualization and accessibility
    fig.update_layout(
        title=dict(
            text="Character Relationship Graph",
            x=0.5,
            font=dict(
                family="Arial", size=20, color="#000000"
            ),  # Larger, high-contrast title
        ),
        font=dict(family="Arial", size=14, color="#000000"),  # Larger, high-contrast font
        showlegend=True,
        hovermode="closest",
        margin=dict(b=30, l=20, r=20, t=50),  # Larger margins for better spacing
        annotations=[
            dict(
                text="Hover over nodes and edges for details. Use toolbar to zoom, pan, or reset view.",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.01,
                y=-0.05,
                font=dict(size=14, color="#000000"),  # High-contrast instruction text
            )
        ],
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1.5, 1.5],  # Wider range for better spread
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1.5, 1.5],  # Wider range for better spread
        ),
        plot_bgcolor="#FFFFFF",  # White background for highest contrast
        height=height,
        width=width,
        legend=dict(
            title=dict(
                text="Relationship Types",
                font=dict(size=16, color="#000000"),  # High-contrast legend title
            ),
            font=dict(size=14, color="#000000"),  # High-contrast legend items
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            itemsizing="constant",
            bordercolor="#000000",  # Black border for legend
            borderwidth=1,
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

    parser = argparse.ArgumentParser(description="Visualize character relationship graph")
    parser.add_argument(
        "--db",
        default=constants.DEFAULT_DB,
        help=f"Path to SQLite database (default: {constants.DEFAULT_DB})",
    )
    parser.add_argument(
        "--output",
        default=constants.DEFAULT_PLOT,
        help=f"Path to save HTML output (default: {constants.DEFAULT_PLOT})",
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
