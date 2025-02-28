"""
Create an accessible and aesthetically pleasing visualization of character relationships.

Features:
- Optimized node separation for maximum readability
- Colorblind-friendly palette
- Distinctive edge styles
- Simple, testable functions with clear docstrings
- Modern Python 3.10+ syntax
"""

# %%
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

from ppcs.constants import Constants

# %%
const = Constants()


def get_db_data(db_path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Extract nodes and edges data from the SQLite database.

    Args:
        db_path: Path to the SQLite database

    Returns:
        Tuple containing lists of node and edge dictionaries

    >>> import tempfile
    >>> with tempfile.NamedTemporaryFile() as f:
    ...     conn = sqlite3.connect(f.name)
    ...     conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, properties TEXT)")
    ...     conn.execute("CREATE TABLE edges (source TEXT, target TEXT, relationship TEXT, weight REAL)")
    ...     conn.execute("INSERT INTO nodes VALUES (?, ?)", ("Character1", '{"id":"Character1","traits":["trait1"]}'))
    ...     conn.execute("INSERT INTO edges VALUES (?, ?, ?, ?)", ("Character1", "Character2", "friend", 0.8))
    ...     conn.commit()
    ...     nodes, edges = get_db_data(f.name)
    >>> len(nodes) > 0 and len(edges) > 0
    True
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get nodes
    cursor = conn.execute("SELECT id, properties FROM nodes")
    nodes = [dict(row) for row in cursor.fetchall()]

    # Get edges
    cursor = conn.execute("SELECT source, target, relationship, weight FROM edges")
    edges = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return nodes, edges


def parse_node_properties(node_data: dict[str, Any]) -> dict[str, Any]:
    """
    Parse node properties from JSON string to dictionary.

    Args:
        node_data: Dictionary with node data including JSON properties

    Returns:
        Dictionary with parsed properties

    >>> data = {"id": "Emma", "properties": '{"id":"Emma","traits":["creative","determined"]}'}
    >>> result = parse_node_properties(data)
    >>> result["id"] == "Emma" and "traits" in result
    True
    """
    try:
        props = json.loads(node_data["properties"])
        return props
    except (json.JSONDecodeError, KeyError):
        return {"id": node_data["id"], "traits": []}


def identify_character_groups(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, int]:
    """
    Group characters based on their relationships for consistent visual representation.

    Args:
        nodes: List of node dictionaries
        edges: List of edge dictionaries

    Returns:
        Dictionary mapping character IDs to group indices

    >>> nodes = [{"id": "Alice"}, {"id": "Bob"}, {"id": "Charlie"}]
    >>> edges = [{"source": "Alice", "target": "Bob"}, {"source": "Charlie", "target": "Alice"}]
    >>> groups = identify_character_groups(nodes, edges)
    >>> isinstance(groups, dict) and all(isinstance(v, int) for v in groups.values())
    True
    """
    # Create undirected graph to find clusters
    G = nx.Graph()

    # Add all nodes
    for node in nodes:
        G.add_node(node["id"])

    # Add edges
    for edge in edges:
        G.add_edge(edge["source"], edge["target"])

    # Find communities using modularity-based detection
    try:
        communities = nx.community.greedy_modularity_communities(G)

        # Map each character to its community
        char_groups = {}
        for i, community in enumerate(communities):
            for character in community:
                char_groups[character] = i

        return char_groups
    except:
        # Fallback to connected components if modularity fails
        components = list(nx.connected_components(G))

        char_groups = {}
        for i, component in enumerate(components):
            for character in component:
                char_groups[character] = i

        return char_groups


def get_colorblind_friendly_palette() -> list[str]:
    """
    Return a colorblind-friendly color palette.

    Returns:
        List of hex color codes

    >>> palette = get_colorblind_friendly_palette()
    >>> len(palette) > 5 and all(c.startswith('#') for c in palette)
    True
    """
    # Wong's colorblind-friendly palette
    return [
        "#000000",  # Black
        "#E69F00",  # Orange
        "#56B4E9",  # Sky Blue
        "#009E73",  # Green
        "#F0E442",  # Yellow
        "#0072B2",  # Blue
        "#D55E00",  # Vermillion
        "#CC79A7",  # Purple
    ]


def get_shape_markers() -> list[str]:
    """
    Return a list of distinctive node shape markers.

    Returns:
        List of matplotlib marker symbols

    >>> markers = get_shape_markers()
    >>> len(markers) > 4
    True
    """
    return ["o", "s", "^", "D", "v", "p", "h", "8"]


def get_edge_styles() -> list[Any]:
    """
    Return a list of distinctive edge line styles.

    Returns:
        List of matplotlib line style specifiers

    >>> styles = get_edge_styles()
    >>> len(styles) > 3
    True
    """
    return ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]


def create_base_graph(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> nx.DiGraph:
    """
    Create the initial NetworkX directed graph with all nodes and edges.

    Args:
        nodes: List of node dictionaries
        edges: List of edge dictionaries

    Returns:
        NetworkX DiGraph object

    >>> nodes = [{"id": "Alice", "properties": '{"id":"Alice","traits":["nice"]}'}]
    >>> edges = [{"source": "Alice", "target": "Bob", "relationship": "friend", "weight": 0.8}]
    >>> G = create_base_graph(nodes, edges)
    >>> len(G.nodes) > 0 and len(G.edges) > 0
    True
    """
    # Create directed graph
    G = nx.DiGraph()

    # Add nodes with properties
    for node in nodes:
        node_id = node["id"]
        properties = parse_node_properties(node)
        traits = properties.get("traits", [])
        G.add_node(node_id, traits=traits)

    # Add edges with attributes
    for edge in edges:
        source = edge["source"]
        target = edge["target"]

        # Skip self-loops
        if source == target:
            continue

        relationship = edge["relationship"]
        weight = edge["weight"]

        # Add edge with attributes
        G.add_edge(source, target, relationship=relationship, weight=weight)

    return G


def optimize_node_layout(G: nx.DiGraph) -> dict[str, np.ndarray]:
    """
    Compute an optimized layout for maximum node separation and readability.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary mapping node IDs to position coordinates

    >>> G = nx.DiGraph()
    >>> G.add_node("A")
    >>> G.add_node("B")
    >>> G.add_edge("A", "B")
    >>> pos = optimize_node_layout(G)
    >>> isinstance(pos, dict) and all(isinstance(v, np.ndarray) for v in pos.values())
    True
    """
    # Try Kamada-Kawai first (good for separation)
    try:
        # Use multiple attempts to find best layout
        best_layout = None
        best_score = -float("inf")

        for i in range(3):
            # Create layout with different seeds
            pos = nx.kamada_kawai_layout(G, scale=2.0)

            # Score the layout based on minimum distance between nodes
            min_dist = float("inf")
            for n1, p1 in pos.items():
                for n2, p2 in pos.items():
                    if n1 != n2:
                        dist = np.sqrt(np.sum((p1 - p2) ** 2))
                        min_dist = min(min_dist, dist)

            # Keep best layout
            if min_dist > best_score:
                best_score = min_dist
                best_layout = pos

        return best_layout
    except:
        # Fallback to spring layout
        return nx.spring_layout(G, k=1.5, iterations=100, seed=42, scale=2.0)


def create_animation_frames(
    G: nx.DiGraph, group_mapping: dict[str, int]
) -> list[tuple[set[str], set[tuple[str, str, str]]]]:
    """
    Create the sequence of frames for the animation.

    Args:
        G: NetworkX graph
        group_mapping: Mapping of nodes to their groups

    Returns:
        List of (nodes, edges) sets for each frame

    >>> G = nx.DiGraph()
    >>> G.add_edge("A", "B", relationship="friend")
    >>> G.add_edge("B", "C", relationship="enemy")
    >>> groups = {"A": 0, "B": 1, "C": 0}
    >>> frames = create_animation_frames(G, groups)
    >>> len(frames) > 0
    True
    """
    # Start with central nodes (higher betweenness)
    try:
        betweenness = nx.betweenness_centrality(G)
        sorted_nodes = sorted(
            G.nodes(), key=lambda n: betweenness.get(n, 0), reverse=True
        )
    except:
        # Fallback to degree if betweenness fails
        sorted_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)

    # Create frames by gradually adding nodes and their edges
    frames = []
    current_nodes = set()
    current_edges = set()

    # First, add top N most central nodes
    central_count = min(3, len(sorted_nodes))
    for i in range(central_count):
        if i < len(sorted_nodes):
            node = sorted_nodes[i]
            current_nodes.add(node)
            frames.append((current_nodes.copy(), current_edges.copy()))

    # Group edges by their source to add coherently
    edges_by_source = defaultdict(list)
    for u, v, data in G.edges(data=True):
        edge_key = (u, v, data["relationship"])
        edges_by_source[u].append((edge_key, v))

    # Then add edges for existing nodes
    for node in list(current_nodes):  # Convert to list for safe iteration
        if node in edges_by_source:
            for edge_key, target in edges_by_source[node]:
                current_edges.add(edge_key)
                current_nodes.add(target)
            frames.append((current_nodes.copy(), current_edges.copy()))

    # Add remaining nodes and edges
    remaining_nodes = [n for n in sorted_nodes if n not in current_nodes]

    for node in remaining_nodes:
        # Add the node
        current_nodes.add(node)
        frames.append((current_nodes.copy(), current_edges.copy()))

        # Add its edges
        if node in edges_by_source:
            for edge_key, target in edges_by_source[node]:
                current_edges.add(edge_key)
                current_nodes.add(target)
            frames.append((current_nodes.copy(), current_edges.copy()))

    # Ensure all edges are included
    for u, v, data in G.edges(data=True):
        edge_key = (u, v, data["relationship"])
        if edge_key not in current_edges:
            current_edges.add(edge_key)
            frames.append((current_nodes.copy(), current_edges.copy()))

    # Add pause frames at the end
    for _ in range(10):
        frames.append((current_nodes.copy(), current_edges.copy()))

    return frames


def draw_frame(
    G: nx.DiGraph,
    ax: plt.Axes,
    pos: dict[str, np.ndarray],
    visible_nodes: set[str],
    visible_edges: set[tuple[str, str, str]],
    node_groups: dict[str, int],
    progress: float = 1.0,
):
    """
    Draw a single frame of the graph visualization.

    Args:
        G: NetworkX graph
        ax: Matplotlib axes
        pos: Node positions
        visible_nodes: Set of visible node IDs
        visible_edges: Set of visible edge keys (source, target, relationship)
        node_groups: Mapping of nodes to their groups
        progress: Animation progress (0.0 to 1.0)

    Returns:
        None
    """
    ax.clear()

    # Setup
    ax.set_title("Character Relationship Network", fontsize=20, fontweight="bold")
    ax.set_axis_off()

    # Get visual elements
    colors = get_colorblind_friendly_palette()
    shapes = get_shape_markers()
    edge_styles = get_edge_styles()

    # Draw only visible nodes
    for node in visible_nodes:
        if node not in G:
            continue

        # Get group for consistent visual attributes
        group = node_groups.get(node, 0)
        color_idx = group % len(colors)
        shape_idx = group % len(shapes)

        # Size based on connections
        size = 1500 + 500 * G.degree(node)

        # Draw the node
        ax.scatter(
            pos[node][0],
            pos[node][1],
            s=size,
            c=colors[color_idx],
            marker=shapes[shape_idx],
            edgecolors="black",
            linewidths=1.5,
            alpha=0.85,
            zorder=3,
        )

        # Draw node label with high contrast
        ax.text(
            pos[node][0],
            pos[node][1],
            node,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9),
            zorder=4,
        )

        # Draw traits below node
        traits = G.nodes[node].get("traits", [])
        if traits:
            traits_str = ", ".join(traits[:3])  # Limit to 3 traits
            ax.text(
                pos[node][0],
                pos[node][1] - 0.08,
                f"Traits: {traits_str}",
                fontsize=8,
                ha="center",
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.2", fc="lightyellow", ec="black", alpha=0.9
                ),
                zorder=3,
            )

    # Draw only visible edges
    visible_edge_styles = {}  # Track used styles for the legend

    for edge_key in visible_edges:
        source, target, relationship = edge_key

        if source not in G or target not in G:
            continue

        # Get edge data
        edge_data = G.get_edge_data(source, target)
        if not edge_data:
            continue

        weight = edge_data.get("weight", 0.5)

        # Assign a consistent style based on relationship
        # Use a hash of the relationship string to pick a style
        style_idx = hash(relationship) % len(edge_styles)
        line_style = edge_styles[style_idx]

        # Track for legend
        visible_edge_styles[relationship] = {
            "style": line_style,
            "width": 1 + 3 * weight,
            "weight": weight,
        }

        # Draw the edge with arrow
        ax.annotate(
            "",
            xy=pos[target],
            xytext=pos[source],
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.1",
                linestyle=line_style,
                linewidth=1 + 3 * weight,
                color="#555555",
                shrinkA=15,
                shrinkB=15,
                alpha=0.8,
            ),
            zorder=1,
        )

        # Edge label with relationship and weight
        midpoint = np.array(pos[source]) * 0.6 + np.array(pos[target]) * 0.4

        # Make edge labels readable with background
        ax.text(
            midpoint[0],
            midpoint[1],
            f"{relationship}\n[{weight:.1f}]",
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
            zorder=2,
        )

    # Build legends
    legend_elements = []

    # Group legend
    used_groups = {node_groups.get(node, 0) for node in visible_nodes if node in G}

    for group in sorted(used_groups):
        if group < len(colors):
            # Find a character from this group for the legend
            chars_in_group = [
                node
                for node in visible_nodes
                if node in G and node_groups.get(node, 0) == group
            ]

            if chars_in_group:
                color_idx = group % len(colors)
                shape_idx = group % len(shapes)

                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker=shapes[shape_idx],
                        color="w",
                        markerfacecolor=colors[color_idx],
                        markeredgecolor="black",
                        markersize=12,
                        label=f"Group: {chars_in_group[0]}",
                    )
                )

    # Relationship style legend (only show top 3 to avoid clutter)
    top_relations = sorted(
        visible_edge_styles.items(), key=lambda x: x[1]["weight"], reverse=True
    )[:3]

    for rel, data in top_relations:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="#555555",
                linestyle=data["style"],
                linewidth=data["width"],
                label=f"{rel} [{data['weight']:.1f}]",
            )
        )

    # Add legend
    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=11,
            frameon=True,
            facecolor="white",
            edgecolor="black",
        )

    # Add progress indicator
    ax.text(
        0.02,
        0.02,
        f"Network: {int(progress * 100)}%",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
    )


def create_character_graph_animation(
    db_path: str, output_path: Path | None, interval: int = 1000
) -> None:
    """
    Create and save an animation of the character relationship graph.

    Args:
        db_path: Path to the SQLite database
        output_path: Path to save the animation
        interval: Milliseconds between frames

    Returns:
        None
    """
    # Get data from database
    nodes, edges = get_db_data(db_path)

    if not nodes or not edges:
        print("No data found in database")
        return

    print(f"Found {len(nodes)} nodes and {len(edges)} edges")

    # Create base graph with all elements
    G = create_base_graph(nodes, edges)

    # Group characters
    character_groups = identify_character_groups(nodes, edges)

    # Calculate optimal layout
    pos = optimize_node_layout(G)

    # Setup figure
    fig, ax = plt.subplots(figsize=(16, 12))
    plt.tight_layout()

    # Create animation frames
    frames = create_animation_frames(G, character_groups)
    total_frames = len(frames)

    # Animation update function
    def update(frame_idx):
        visible_nodes, visible_edges = frames[frame_idx]
        progress = (frame_idx + 1) / total_frames
        draw_frame(G, ax, pos, visible_nodes, visible_edges, character_groups, progress)

    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=interval, blit=False, repeat=False
    )

    # Save animation
    if output_path:
        print(f"Saving animation to {output_path}...")

        # Determine format based on extension
        extension = output_path.suffix.lower()

        if extension == ".gif":
            # For GIF
            anim.save(output_path, writer="pillow", fps=2, dpi=100)
        elif extension in [".mp4", ".avi", ".mov"]:
            # For video formats
            writer = animation.FFMpegWriter(
                fps=2, metadata=dict(title="Character Relationship Network"), bitrate=1800
            )
            anim.save(output_path, writer=writer)
        else:
            # Default to MP4
            output_path = output_path.with_suffix(".mp4")
            writer = animation.FFMpegWriter(
                fps=2, metadata=dict(title="Character Relationship Network"), bitrate=1800
            )
            anim.save(output_path, writer=writer)

        print(f"Animation saved to {output_path}")

    plt.show()


def main() -> None:
    """
    Main function to run the character graph visualization.
    """
    # Database path
    db_path = const.DEFAULT_DB

    # Output path
    output_path = const.DEFAULT_ANIMATION

    # Create and save animation
    create_character_graph_animation(
        db_path=db_path,
        output_path=output_path,
        interval=1200,  # Slower animation for better readability
    )


if __name__ == "__main__":
    main()
