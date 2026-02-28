"""
AceML Studio – Knowledge Graph Engine
========================================
Builds entity-relationship graphs from tabular data and provides:

  • Schema-level graph  — columns as nodes, edges weighted by correlation /
                          categorical association (Cramér's V)
  • Entity-level graph  — unique values in key columns as nodes, connected
                          by co-occurrence in the same row
  • Graph metrics       — degree, betweenness centrality, PageRank, clustering
  • Community detection — Louvain-style greedy modularity (pure numpy/scipy)
  • Export              — node-link JSON compatible with D3 / Cytoscape / vis.js

Optional:
  • networkx ≥ 3.0 — richer algorithms (PageRank, betweenness, Louvain)
  • scipy ≥ 1.11   — chi-squared test for categorical associations
"""

import logging
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("aceml.knowledge_graph")

# ── Optional imports ──────────────────────────────────────────────
try:
    import networkx as nx  # type: ignore
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logger.info("networkx not installed – using pure-numpy graph algorithms")

try:
    from scipy.stats import chi2_contingency  # type: ignore
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.info("scipy not installed – using basic association measures")


# ════════════════════════════════════════════════════════════════════
#  Knowledge Graph Engine
# ════════════════════════════════════════════════════════════════════

class KnowledgeGraphEngine:
    """
    Builds and analyses knowledge graphs from tabular ML datasets.

    Two graph modes:
      schema_graph  – columns as nodes, edges by statistical association
      entity_graph  – unique values as nodes, edges by row co-occurrence
    """

    # ----------------------------------------------------------------
    #  Schema Graph
    # ----------------------------------------------------------------
    @staticmethod
    def build_schema_graph(
        df: pd.DataFrame,
        correlation_threshold: float = 0.3,
        include_categorical: bool = True,
    ) -> Dict[str, Any]:
        """
        Build a column-level relationship graph.

        Edges:
          numeric–numeric  →  Pearson correlation
          cat–cat          →  Cramér's V (association)
          numeric–cat      →  point-biserial correlation (absolute)
        """
        start_time = time.time()
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        dt_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
        all_cols = df.columns.tolist()

        # ── Build nodes ──────────────────────────────────────────────
        for col in all_cols:
            series = df[col]
            dtype_group = (
                "numeric" if col in numeric_cols
                else "categorical" if col in cat_cols
                else "datetime" if col in dt_cols
                else "other"
            )
            null_pct = round(float(series.isna().mean() * 100), 2)
            node: Dict[str, Any] = {
                "id": col,
                "label": col,
                "dtype_group": dtype_group,
                "dtype": str(series.dtype),
                "null_pct": null_pct,
                "unique": int(series.nunique()),
                "size": max(10, 40 - int(null_pct / 5)),  # visual hint
            }
            if col in numeric_cols:
                node["mean"] = round(float(series.mean()), 4)
                node["std"] = round(float(series.std()), 4)
            else:
                top = series.value_counts().index[:3].tolist()
                node["top_values"] = [str(v) for v in top]
            nodes.append(node)

        # ── Numeric–numeric edges (Pearson) ──────────────────────────
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    v = float(corr_matrix.iloc[i, j])  # type: ignore[arg-type]
                    if abs(v) >= correlation_threshold:
                        edges.append({
                            "id": f"{numeric_cols[i]}__{numeric_cols[j]}",
                            "source": numeric_cols[i],
                            "target": numeric_cols[j],
                            "weight": round(abs(v), 4),
                            "signed_weight": round(v, 4),
                            "type": "pearson_correlation",
                            "label": f"r={v:.2f}",
                        })

        # ── Categorical–categorical edges (Cramér's V) ───────────────
        if include_categorical and len(cat_cols) >= 2:
            for i in range(len(cat_cols)):
                for j in range(i + 1, len(cat_cols)):
                    v = KnowledgeGraphEngine._cramers_v(df[cat_cols[i]], df[cat_cols[j]])
                    if v is not None and v >= correlation_threshold:
                        edges.append({
                            "id": f"{cat_cols[i]}__{cat_cols[j]}",
                            "source": cat_cols[i],
                            "target": cat_cols[j],
                            "weight": round(v, 4),
                            "signed_weight": round(v, 4),
                            "type": "cramers_v",
                            "label": f"V={v:.2f}",
                        })

        # ── Numeric–categorical edges (eta-squared proxy) ────────────
        if include_categorical:
            for num_col in numeric_cols:
                for cat_col in cat_cols:
                    v = KnowledgeGraphEngine._eta_squared(df[num_col], df[cat_col])
                    if v is not None and v >= correlation_threshold:
                        edges.append({
                            "id": f"{num_col}__{cat_col}",
                            "source": num_col,
                            "target": cat_col,
                            "weight": round(v, 4),
                            "signed_weight": round(v, 4),
                            "type": "eta_squared",
                            "label": f"η²={v:.2f}",
                        })

        duration = round(time.time() - start_time, 3)
        return {
            "graph_type": "schema_graph",
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "nodes": nodes,
            "edges": edges,
            "correlation_threshold": correlation_threshold,
            "duration_sec": duration,
        }

    # ----------------------------------------------------------------
    #  Entity Graph
    # ----------------------------------------------------------------
    @staticmethod
    def build_entity_graph(
        df: pd.DataFrame,
        key_columns: Optional[List[str]] = None,
        max_unique_per_col: int = 50,
        min_cooccurrences: int = 2,
    ) -> Dict[str, Any]:
        """
        Build a value-level co-occurrence graph.

        Nodes = unique values in key_columns.
        Edges = two values appeared in the same row (weighted by frequency).
        """
        start_time = time.time()

        # Auto-select low-cardinality categorical columns if none specified
        if key_columns is None:
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            key_columns = [c for c in cat_cols if df[c].nunique() <= max_unique_per_col][:8]

        if not key_columns:
            return {
                "graph_type": "entity_graph",
                "error": "No suitable categorical columns found for entity graph",
                "n_nodes": 0,
                "n_edges": 0,
            }

        # Build node registry and edge accumulator
        node_freq: Dict[str, int] = defaultdict(int)
        node_col: Dict[str, str] = {}
        cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)

        for _, row in df[key_columns].iterrows():
            row_vals = [(col, str(row[col])) for col in key_columns if pd.notna(row[col])]
            # Count individual values
            for col, val in row_vals:
                key = f"{col}::{val}"
                node_freq[key] += 1
                node_col[key] = col

            # Count co-occurrences (all pairs)
            for i in range(len(row_vals)):
                for j in range(i + 1, len(row_vals)):
                    col_i, val_i = row_vals[i]
                    col_j, val_j = row_vals[j]
                    n_i = f"{col_i}::{val_i}"
                    n_j = f"{col_j}::{val_j}"
                    edge_key = (min(n_i, n_j), max(n_i, n_j))
                    cooccurrence[edge_key] += 1

        # Build nodes
        nodes = [
            {
                "id": node_id,
                "label": node_id.split("::", 1)[1],
                "column": node_col[node_id],
                "frequency": node_freq[node_id],
                "size": max(5, min(40, node_freq[node_id] // 10 + 5)),
            }
            for node_id in node_freq
        ]

        # Build edges (filtered by min_cooccurrences)
        edges = [
            {
                "id": f"{src}__{tgt}",
                "source": src,
                "target": tgt,
                "weight": count,
                "label": str(count),
                "type": "co_occurrence",
            }
            for (src, tgt), count in cooccurrence.items()
            if count >= min_cooccurrences
        ]

        duration = round(time.time() - start_time, 3)
        return {
            "graph_type": "entity_graph",
            "key_columns": key_columns,
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "nodes": nodes,
            "edges": edges,
            "min_cooccurrences": min_cooccurrences,
            "duration_sec": duration,
        }

    # ----------------------------------------------------------------
    #  Graph Metrics
    # ----------------------------------------------------------------
    @staticmethod
    def compute_graph_metrics(graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute graph-level and node-level metrics.

        If networkx is available, uses full algorithms.
        Falls back to degree-only metrics otherwise.
        """
        start_time = time.time()
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        if not nodes:
            return {"error": "Graph has no nodes"}

        node_ids = [n["id"] for n in nodes]
        node_idx = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)

        # ── Degree (always computable) ────────────────────────────────
        degree: Dict[str, int] = defaultdict(int)
        weighted_degree: Dict[str, float] = defaultdict(float)
        for e in edges:
            src, tgt = e["source"], e["target"]
            w = float(e.get("weight", 1.0))
            degree[src] += 1
            degree[tgt] += 1
            weighted_degree[src] += w
            weighted_degree[tgt] += w

        node_metrics = {
            nid: {
                "degree": degree[nid],
                "weighted_degree": round(weighted_degree[nid], 4),
            }
            for nid in node_ids
        }

        # ── NetworkX-powered metrics ─────────────────────────────────
        if HAS_NETWORKX:
            import networkx as nx
            G = nx.Graph()
            for n_node in nodes:
                G.add_node(n_node["id"])
            for e in edges:
                G.add_edge(e["source"], e["target"], weight=e.get("weight", 1.0))

            # Degree centrality
            dc = nx.degree_centrality(G)
            for nid in node_ids:
                node_metrics[nid]["degree_centrality"] = round(dc.get(nid, 0.0), 4)

            # Betweenness centrality (sampled for large graphs)
            try:
                if n > 500:
                    bc = nx.betweenness_centrality(G, k=min(100, n), weight="weight")
                else:
                    bc = nx.betweenness_centrality(G, weight="weight")
                for nid in node_ids:
                    node_metrics[nid]["betweenness_centrality"] = round(bc.get(nid, 0.0), 6)
            except Exception:
                pass

            # PageRank
            try:
                pr = nx.pagerank(G, weight="weight")
                for nid in node_ids:
                    node_metrics[nid]["pagerank"] = round(pr.get(nid, 0.0), 6)
            except Exception:
                pass

            # Clustering coefficient
            try:
                cc = nx.clustering(G, weight="weight")
                cc_dict: Dict[str, float] = dict(cc)  # type: ignore[arg-type]
                for nid in node_ids:
                    node_metrics[nid]["clustering"] = round(cc_dict.get(nid, 0.0), 4)
            except Exception:
                pass

            # Graph-level metrics
            n_components = nx.number_connected_components(G)
            density = round(nx.density(G), 4)
            degree_dict: Dict[str, int] = dict(G.degree())  # type: ignore[arg-type]
            avg_degree = round(sum(degree_dict.values()) / max(n, 1), 4)
            try:
                avg_clustering = round(float(nx.average_clustering(G)), 4)
            except Exception:
                avg_clustering = None
            try:
                lcc = max(nx.connected_components(G), key=len)
                avg_shortest_path = round(
                    float(nx.average_shortest_path_length(G.subgraph(lcc))), 4
                ) if len(lcc) > 1 else None
            except Exception:
                avg_shortest_path = None

        else:
            # Pure-numpy fallback
            density = round(2 * len(edges) / max(n * (n - 1), 1), 4)
            avg_degree = round(sum(degree.values()) / max(n, 1), 4)
            n_components = 1  # approximate
            avg_clustering = None
            avg_shortest_path = None

        # Top nodes by degree
        top_nodes = sorted(node_ids, key=lambda x: degree[x], reverse=True)[:10]

        # Attach metrics back to node list
        enriched_nodes = [
            {**nd, **node_metrics.get(nd["id"], {})}
            for nd in nodes
        ]

        duration = round(time.time() - start_time, 3)
        return {
            "n_nodes": n,
            "n_edges": len(edges),
            "graph_density": density,
            "avg_degree": avg_degree,
            "n_connected_components": n_components,
            "avg_clustering_coefficient": avg_clustering,
            "avg_shortest_path": avg_shortest_path,
            "top_nodes_by_degree": top_nodes,
            "node_metrics": node_metrics,
            "enriched_nodes": enriched_nodes,
            "uses_networkx": HAS_NETWORKX,
            "duration_sec": duration,
        }

    # ----------------------------------------------------------------
    #  Community Detection
    # ----------------------------------------------------------------
    @staticmethod
    def detect_communities(graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect communities (clusters) in the graph.
        Uses networkx greedy modularity if available, else degree-based heuristic.
        """
        start_time = time.time()
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        node_ids = [n["id"] for n in nodes]

        if not node_ids:
            return {"error": "Graph has no nodes"}

        if HAS_NETWORKX:
            import networkx as nx
            G = nx.Graph()
            for n_node in nodes:
                G.add_node(n_node["id"])
            for e in edges:
                G.add_edge(e["source"], e["target"], weight=e.get("weight", 1.0))

            try:
                from networkx.algorithms.community import greedy_modularity_communities
                communities_raw = list(greedy_modularity_communities(G, weight="weight"))
                communities = [sorted(list(c)) for c in communities_raw]
            except Exception:
                # Fallback: connected components as communities
                communities = [sorted(list(c)) for c in nx.connected_components(G)]

            # Modularity score
            try:
                from networkx.algorithms.community.quality import modularity
                community_sets = [set(c) for c in communities]
                mod_score = round(float(modularity(G, community_sets)), 4)
            except Exception:
                mod_score = None

        else:
            # Simple heuristic: group by shared neighbourhood (degree-based)
            communities = KnowledgeGraphEngine._simple_communities(node_ids, edges)
            mod_score = None

        # Assign community ids to nodes
        node_community: Dict[str, int] = {}
        for cid, community in enumerate(communities):
            for nid in community:
                node_community[nid] = cid

        enriched_nodes = [
            {**n_node, "community": node_community.get(n_node["id"], 0)}
            for n_node in nodes
        ]

        duration = round(time.time() - start_time, 3)
        return {
            "n_communities": len(communities),
            "modularity": mod_score,
            "communities": [
                {"community_id": cid, "members": members, "size": len(members)}
                for cid, members in enumerate(communities)
            ],
            "node_community_map": node_community,
            "enriched_nodes": enriched_nodes,
            "algorithm": "greedy_modularity" if HAS_NETWORKX else "simple_heuristic",
            "duration_sec": duration,
        }

    # ----------------------------------------------------------------
    #  Node Neighbours
    # ----------------------------------------------------------------
    @staticmethod
    def node_neighbors(
        graph_data: Dict[str, Any],
        node_id: str,
        depth: int = 1,
    ) -> Dict[str, Any]:
        """Return the neighbourhood subgraph up to `depth` hops."""
        edges = graph_data.get("edges", [])
        nodes_dict = {n["id"]: n for n in graph_data.get("nodes", [])}

        if node_id not in nodes_dict:
            return {"error": f"Node '{node_id}' not found"}

        # Build adjacency
        adj: Dict[str, List[str]] = defaultdict(list)
        edge_map: Dict[Tuple[str, str], Dict] = {}
        for e in edges:
            adj[e["source"]].append(e["target"])
            adj[e["target"]].append(e["source"])
            edge_map[(e["source"], e["target"])] = e
            edge_map[(e["target"], e["source"])] = e

        # BFS up to `depth`
        visited = {node_id}
        frontier = {node_id}
        for _ in range(depth):
            new_frontier: set = set()
            for nid in frontier:
                for neighbor in adj[nid]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_frontier.add(neighbor)
            frontier = new_frontier

        subgraph_nodes = [nodes_dict[nid] for nid in visited if nid in nodes_dict]
        subgraph_edges = [
            e for e in edges
            if e["source"] in visited and e["target"] in visited
        ]

        return {
            "node_id": node_id,
            "depth": depth,
            "n_neighbors": len(visited) - 1,
            "subgraph_nodes": subgraph_nodes,
            "subgraph_edges": subgraph_edges,
        }

    # ----------------------------------------------------------------
    #  Export (node-link JSON)
    # ----------------------------------------------------------------
    @staticmethod
    def export_graph(
        graph_data: Dict[str, Any],
        fmt: str = "node_link",
    ) -> Dict[str, Any]:
        """
        Export graph in a format suitable for front-end rendering.
        fmt: 'node_link' (D3/vis.js), 'cytoscape', 'adjacency_matrix'
        """
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        if fmt == "node_link":
            return {
                "format": "node_link",
                "nodes": nodes,
                "links": edges,
                "metadata": {
                    "n_nodes": len(nodes),
                    "n_edges": len(edges),
                    "graph_type": graph_data.get("graph_type", "unknown"),
                },
            }

        elif fmt == "cytoscape":
            elements: List[Dict] = []
            for n_node in nodes:
                elements.append({
                    "data": {
                        "id": n_node["id"],
                        "label": n_node.get("label", n_node["id"]),
                        **{k: v for k, v in n_node.items() if k not in ("id", "label")},
                    },
                    "group": "nodes",
                })
            for e in edges:
                elements.append({
                    "data": {
                        "id": e.get("id", f"{e['source']}__{e['target']}"),
                        "source": e["source"],
                        "target": e["target"],
                        "weight": e.get("weight", 1.0),
                        "label": e.get("label", ""),
                    },
                    "group": "edges",
                })
            return {"format": "cytoscape", "elements": elements}

        elif fmt == "adjacency_matrix":
            node_ids = [n["id"] for n in nodes]
            idx = {nid: i for i, nid in enumerate(node_ids)}
            n = len(node_ids)
            matrix = [[0.0] * n for _ in range(n)]
            for e in edges:
                i, j = idx.get(e["source"], -1), idx.get(e["target"], -1)
                if i >= 0 and j >= 0:
                    w = float(e.get("weight", 1.0))
                    matrix[i][j] = w
                    matrix[j][i] = w
            return {
                "format": "adjacency_matrix",
                "node_ids": node_ids,
                "matrix": matrix,
            }

        return {"error": f"Unknown export format: {fmt}"}

    # ----------------------------------------------------------------
    #  Utilities / Helpers
    # ----------------------------------------------------------------
    @staticmethod
    def _cramers_v(col_a: pd.Series, col_b: pd.Series) -> Optional[float]:
        """Cramér's V association measure between two categorical columns."""
        try:
            combined = pd.DataFrame({"a": col_a, "b": col_b}).dropna()
            if len(combined) < 10:
                return None
            ct = pd.crosstab(combined["a"], combined["b"])
            if HAS_SCIPY:
                from scipy.stats import chi2_contingency
                chi2, _, _, _ = chi2_contingency(ct)
            else:
                # Manual chi2
                observed = ct.values.astype(float)
                row_sums = observed.sum(axis=1, keepdims=True)
                col_sums = observed.sum(axis=0, keepdims=True)
                total = observed.sum()
                expected = row_sums @ col_sums / total
                with np.errstate(divide="ignore", invalid="ignore"):
                    chi2 = float(np.nansum((observed - expected) ** 2 / np.where(expected == 0, np.inf, expected)))
            n = ct.values.sum()
            min_dim = min(ct.shape) - 1
            if min_dim <= 0 or n <= 0:
                return None
            v = float(np.sqrt(chi2 / (n * max(min_dim, 1))))
            return round(min(v, 1.0), 4)
        except Exception:
            return None

    @staticmethod
    def _eta_squared(numeric_col: pd.Series, cat_col: pd.Series) -> Optional[float]:
        """Eta-squared (effect size) for numeric–categorical relationship."""
        try:
            combined = pd.DataFrame({"num": numeric_col, "cat": cat_col}).dropna()
            if len(combined) < 10:
                return None
            groups_np: List[np.ndarray] = [
                np.array(group["num"].values, dtype=float)
                for _, group in combined.groupby("cat")
            ]
            if len(groups_np) < 2:
                return None
            overall_mean = float(combined["num"].mean())  # type: ignore[arg-type]
            ss_between = float(sum(
                len(g) * (g.mean() - overall_mean) ** 2 for g in groups_np
            ))
            num_arr = np.array(combined["num"].values, dtype=float)
            ss_total = float(num_arr.var() * (len(combined) - 1))
            if ss_total <= 0:
                return None
            return round(min(ss_between / ss_total, 1.0), 4)
        except Exception:
            return None

    @staticmethod
    def _simple_communities(
        node_ids: List[str], edges: List[Dict]
    ) -> List[List[str]]:
        """
        Greedy community detection fallback (union-find style).
        Groups nodes that share at least one edge.
        """
        parent: Dict[str, str] = {n: n for n in node_ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            parent[find(x)] = find(y)

        for e in edges:
            src, tgt = e.get("source", ""), e.get("target", "")
            if src in parent and tgt in parent:
                union(src, tgt)

        groups: Dict[str, List[str]] = defaultdict(list)
        for nid in node_ids:
            groups[find(nid)].append(nid)

        return [sorted(members) for members in groups.values()]
