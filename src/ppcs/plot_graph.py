import json

import networkx as nx
import plotly.graph_objects as go
from plotly.offline import plot

from ppcs.main import logger


# %%
def plot_character_graph(
    db_path: str,
    output_path: str | None = None,
    min_weight: float = 0.0,
    max_connections: int | None = None,
    open_browser: bool = True,
) -> go.Figure:
    """
    Generate an interactive graph visualization of character relationships.

    This function reads from the SQLite database and creates a NetworkX graph,
    then visualizes it using Plotly for an interactive experience.

    Args:
        db_path: Path to SQLite database
        output_path: Path to save HTML output (if None, only returns the figure)
        min_weight: Minimum relationship weight to include (0.0-1.0)
        max_connections: Limit number of connections per node for readability
        open_browser: Whether to open the visualization in the browser

    Returns:
        Plotly Figure object

    Examples:
        >>> fig = plot_character_graph("data/world.db", "character_network.html", min_weight=0.3)
        >>> isinstance(fig, go.Figure)
        True
    """
    # Step 1: Build graph from database
    G, node_info = build_graph_from_db(db_path, min_weight, max_connections)

    if not G.nodes():
        logger.warning("No nodes found in the database with the current filters")
        return go.Figure()

    # Step 2: Compute layout - Force-directed algorithm works well for relationship networks
    pos = nx.spring_layout(G, seed=42, k=0.3)  # k controls node spacing

    # Step 3: Create node traces with different colors by community
    communities = detect_communities(G)
    node_traces = create_node_traces(G, pos, communities, node_info)

    # Step 4: Create edge traces with varying thickness by weight
    edge_trace = create_edge_trace(G, pos)

    # Step 5: Create the figure
    fig = go.Figure(
        data=node_traces + [edge_trace],
        layout=go.Layout(
            title="Character Relationship Network",
            titlefont=dict(size=16),
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white",
            legend=dict(
                title="Character Communities",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
            ),
        ),
    )

    # Add annotations for edges to display relationship types
    add_edge_annotations(fig, G, pos)

    # Save to HTML if output path provided
    if output_path:
        plot(fig, filename=output_path, auto_open=open_browser)
        logger.info(f"Graph saved to {output_path}")

    return fig


# %%
def build_graph_from_db(
    db_path: str, min_weight: float = 0.0, max_connections: int | None = None
) -> tuple[nx.Graph, dict]:
    """
    Build a NetworkX graph from the SQLite database.

    Args:
        db_path: Path to SQLite database
        min_weight: Minimum relationship weight to include
        max_connections: Maximum connections per node (for readability)

    Returns:
        tuple containing (NetworkX graph, node information dictionary)
    """
    import sqlite3

    G = nx.Graph()
    node_info = {}

    with sqlite3.connect(db_path) as conn:
        # Get nodes
        cursor = conn.execute("SELECT id, properties FROM nodes")
        for node_id, properties_json in cursor.fetchall():
            properties = json.loads(properties_json)
            G.add_node(node_id)
            node_info[node_id] = properties

        # Get edges with weight filter
        cursor = conn.execute(
            "SELECT source, target, relationship, weight FROM edges WHERE weight >= ?",
            (min_weight,),
        )

        # Process all edges
        edges = [
            (source, target, {"relationship": rel, "weight": weight})
            for source, target, rel, weight in cursor.fetchall()
        ]

        # If max_connections is set, limit edges per node
        if max_connections:
            # Sort edges by weight
            edges.sort(key=lambda x: x[2]["weight"], reverse=True)

            # Count connections per node
            node_connections = {node: 0 for node in G.nodes()}

            # Only add edges if both nodes haven't exceeded max connections
            filtered_edges = []
            for source, target, attrs in edges:
                if (
                    node_connections.get(source, 0) < max_connections
                    and node_connections.get(target, 0) < max_connections
                ):
                    filtered_edges.append((source, target, attrs))
                    node_connections[source] = node_connections.get(source, 0) + 1
                    node_connections[target] = node_connections.get(target, 0) + 1

            edges = filtered_edges

        # Add edges to graph
        G.add_edges_from(edges)

    return G, node_info


# %%
def detect_communities(G: nx.Graph) -> dict[str, int]:
    """
    Detect communities within the graph for coloring.

    Args:
        G: NetworkX graph

    Returns:
        dictionary mapping node IDs to community numbers
    """
    # Use Louvain method for community detection (often works well for relationship graphs)
    try:
        from community import best_partition

        return best_partition(G)
    except ImportError:
        # Fallback to connected components if python-louvain not available
        logger.warning("python-louvain not installed; using connected components instead")
        communities = {}
        for i, component in enumerate(nx.connected_components(G)):
            for node in component:
                communities[node] = i
        return communities


# %%
def create_node_traces(
    G: nx.Graph, pos: dict, communities: dict[str, int], node_info: dict
) -> list[go.Scatter]:
    """
    Create Plotly traces for nodes, colored by community.

    Args:
        G: NetworkX graph
        pos: Node positions
        communities: Community assignments
        node_info: Node information from database

    Returns:
        list of Plotly Scatter traces for nodes
    """
    # Get unique community ids
    community_ids = sorted(set(communities.values()))

    # Create a trace for each community
    node_traces = []

    # Color palette - colorblind-friendly
    colors = [
        "#1F77B4",
        "#FF7F0E",
        "#2CA02C",
        "#D62728",
        "#9467BD",
        "#8C564B",
        "#E377C2",
        "#7F7F7F",
        "#BCBD22",
        "#17BECF",
    ]

    for community_id in community_ids:
        # Get nodes in this community
        community_nodes = [
            node for node in G.nodes() if communities.get(node) == community_id
        ]

        if not community_nodes:
            continue

        x = [pos[node][0] for node in community_nodes]
        y = [pos[node][1] for node in community_nodes]

        # Node size based on degree centrality (more connections = larger node)
        node_degrees = [G.degree(node) * 5 + 15 for node in community_nodes]

        # Prepare hover text with node information
        hover_texts = []
        for node in community_nodes:
            traits = node_info.get(node, {}).get("traits", [])
            traits_str = ", ".join(traits) if traits else "None"
            connections = ", ".join(neighbor for neighbor in G.neighbors(node))

            hover_text = (
                f"<b>{node}</b><br>"
                f"Traits: {traits_str}<br>"
                f"Connections: {G.degree(node)}<br>"
                f"Connected to: {connections}"
            )
            hover_texts.append(hover_text)

        # Create a trace for this community
        trace = go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=node_degrees,
                color=colors[community_id % len(colors)],
                line=dict(width=2, color="white"),
            ),
            text=hover_texts,
            hoverinfo="text",
            name=f"Community {community_id}",
        )

        node_traces.append(trace)

    return node_traces


# %%
def create_edge_trace(G: nx.Graph, pos: dict) -> go.Scatter:
    """
    Create Plotly trace for edges with varying thickness.

    Args:
        G: NetworkX graph
        pos: Node positions

    Returns:
        Plotly Scatter trace for edges
    """
    edge_x = []
    edge_y = []
    edge_weights = []
    edge_info = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        # Add line coordinates
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        # Add edge data for hover
        weight = edge[2].get("weight", 0.5)
        relationship = edge[2].get("relationship", "connected")

        edge_weights.append(weight)
        edge_info.append(
            f"{edge[0]} -- {relationship} --> {edge[1]} (strength: {weight:.2f})"
        )

    # Create colorscale for edge weights
    edge_colors = []
    for i in range(0, len(edge_x), 3):
        idx = i // 3
        if idx < len(edge_weights):
            # Convert weight (0-1) to a color using a gradient
            # High weight = stronger connection = darker color
            edge_colors.extend([edge_weights[idx], edge_weights[idx], None])

    return go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(
            width=1,
            color="rgba(150,150,150,0.5)",
        ),
        hoverinfo="none",
        mode="lines",
        name="Relationships",
        showlegend=False,
    )


# %%
def add_edge_annotations(fig: go.Figure, G: nx.Graph, pos: dict) -> None:
    """
    Add annotations to show relationship types on hover.

    Args:
        fig: Plotly figure
        G: NetworkX graph
        pos: Node positions
    """
    annotations = []

    # Add invisible hover points at the middle of each edge for relationship info
    for source, target, data in G.edges(data=True):
        x0, y0 = pos[source]
        x1, y1 = pos[target]

        # Calculate midpoint for the annotation position
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2

        # Get relationship information
        relationship = data.get("relationship", "connected")
        weight = data.get("weight", 0.5)

        # Add annotation
        annotations.append(
            dict(
                x=mid_x,
                y=mid_y,
                xref="x",
                yref="y",
                text=f"{relationship} ({weight:.2f})",
                showarrow=False,
                font=dict(family="Arial", size=8, color="rgba(50,50,50,0.8)"),
                bordercolor="rgba(0,0,0,0)",
                borderwidth=1,
                borderpad=2,
                bgcolor="rgba(255,255,255,0.7)",
                opacity=0.7,
            )
        )

    fig.update_layout(annotations=annotations)


# %%
if __name__ == "__main__":
    # Example usage
    db_path = "data/graph_objects.db"
    output_path = "character_network.html"

    fig = plot_character_graph(
        db_path=db_path,
        output_path=output_path,
        min_weight=0.2,
        max_connections=10,
        open_browser=True,
    )
