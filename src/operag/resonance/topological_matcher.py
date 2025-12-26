"""Topological Resonance - Computation as Topological Pattern Matching.

This module implements the core concept: computation == topological resonance.
It performs pattern matching in topological spaces, where compatible structures
resonate and interact.
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
from ..membranes.prime_membranes import PrimeFactorMembrane
from ..hypergraph.tensor_embeddings import HypergraphTensorEmbedding
from ..operad.gadget_constellation import OperadGadget, GadgetConstellation


class TopologicalMatcher:
    """Performs topological matching between computational structures.
    
    The matcher identifies resonance between different topological representations:
    - Membrane topologies
    - Hypergraph embeddings
    - Operad constellations
    
    Resonance occurs when topological signatures align.
    """
    
    def __init__(self, threshold: float = 0.7):
        """Initialize the topological matcher.
        
        Args:
            threshold: Similarity threshold for resonance (0-1)
        """
        self.threshold = threshold
        self.resonance_cache: Dict[Tuple, float] = {}
        
    def membrane_resonance(self, 
                          m1: PrimeFactorMembrane, 
                          m2: PrimeFactorMembrane) -> float:
        """Compute resonance between two membranes based on topology.
        
        Args:
            m1: First membrane
            m2: Second membrane
            
        Returns:
            Resonance score (0-1)
        """
        sig1 = set(m1.get_topology_signature())
        sig2 = set(m2.get_topology_signature())
        
        if not sig1 and not sig2:
            return 1.0
        
        if not sig1 or not sig2:
            return 0.0
        
        # Jaccard similarity of prime factors
        intersection = len(sig1 & sig2)
        union = len(sig1 | sig2)
        
        return intersection / union if union > 0 else 0.0
    
    def hypergraph_resonance(self,
                            h1: HypergraphTensorEmbedding,
                            h2: HypergraphTensorEmbedding) -> float:
        """Compute resonance between two hypergraphs.
        
        Args:
            h1: First hypergraph
            h2: Second hypergraph
            
        Returns:
            Resonance score (0-1)
        """
        sig1 = h1.get_topology_signature()
        sig2 = h2.get_topology_signature()
        
        # Compare structural properties
        node_ratio = min(sig1["num_nodes"], sig2["num_nodes"]) / max(sig1["num_nodes"], sig2["num_nodes"])
        edge_ratio = min(sig1["num_edges"], sig2["num_edges"]) / max(sig1["num_edges"], sig2["num_edges"])
        
        # Compare degree distributions
        degree_diff = abs(sig1["avg_degree"] - sig2["avg_degree"])
        max_degree = max(sig1["max_degree"], sig2["max_degree"])
        degree_similarity = 1.0 - (degree_diff / max_degree if max_degree > 0 else 0)
        
        # Weighted combination
        resonance = 0.3 * node_ratio + 0.3 * edge_ratio + 0.4 * degree_similarity
        
        return float(np.clip(resonance, 0, 1))
    
    def constellation_resonance(self,
                               c1: GadgetConstellation,
                               c2: GadgetConstellation) -> float:
        """Compute resonance between two gadget constellations.
        
        Args:
            c1: First constellation
            c2: Second constellation
            
        Returns:
            Resonance score (0-1)
        """
        sig1 = c1.get_topology_signature()
        sig2 = c2.get_topology_signature()
        
        # Compare structural properties
        gadget_ratio = min(sig1["num_gadgets"], sig2["num_gadgets"]) / max(sig1["num_gadgets"], sig2["num_gadgets"])
        conn_ratio = min(sig1["num_connections"], sig2["num_connections"]) / max(sig1["num_connections"], sig2["num_connections"])
        
        # Compare gadget type overlap
        types1 = sig1["gadget_types"]
        types2 = sig2["gadget_types"]
        type_overlap = len(types1 & types2) / len(types1 | types2) if (types1 | types2) else 1.0
        
        # Weighted combination
        resonance = 0.3 * gadget_ratio + 0.3 * conn_ratio + 0.4 * type_overlap
        
        return float(np.clip(resonance, 0, 1))
    
    def cross_topology_resonance(self,
                                obj1: Any,
                                obj2: Any) -> float:
        """Compute resonance between objects of different topological types.
        
        Args:
            obj1: First object
            obj2: Second object
            
        Returns:
            Resonance score (0-1)
        """
        # Membrane to Hypergraph
        if isinstance(obj1, PrimeFactorMembrane) and isinstance(obj2, HypergraphTensorEmbedding):
            depth = obj1.get_depth()
            nodes = obj2.num_nodes
            # Compare structural complexity
            complexity_ratio = min(depth, nodes) / max(depth, nodes) if max(depth, nodes) > 0 else 0
            return float(complexity_ratio)
        
        # Hypergraph to Constellation
        if isinstance(obj1, HypergraphTensorEmbedding) and isinstance(obj2, GadgetConstellation):
            edges = len(obj1.hyperedges)
            gadgets = len(obj2.gadgets)
            complexity_ratio = min(edges, gadgets) / max(edges, gadgets) if max(edges, gadgets) > 0 else 0
            return float(complexity_ratio)
        
        # Membrane to Constellation
        if isinstance(obj1, PrimeFactorMembrane) and isinstance(obj2, GadgetConstellation):
            depth = obj1.get_depth()
            const_depth = obj2.compute_depth()
            depth_ratio = min(depth, const_depth) / max(depth, const_depth) if max(depth, const_depth) > 0 else 0
            return float(depth_ratio)
        
        # Reverse cases
        if isinstance(obj2, PrimeFactorMembrane) or isinstance(obj2, HypergraphTensorEmbedding) or isinstance(obj2, GadgetConstellation):
            return self.cross_topology_resonance(obj2, obj1)
        
        return 0.0
    
    def compute_resonance(self, obj1: Any, obj2: Any) -> float:
        """Compute topological resonance between any two objects.
        
        Args:
            obj1: First object
            obj2: Second object
            
        Returns:
            Resonance score (0-1)
        """
        # Same type resonances
        if isinstance(obj1, PrimeFactorMembrane) and isinstance(obj2, PrimeFactorMembrane):
            return self.membrane_resonance(obj1, obj2)
        
        if isinstance(obj1, HypergraphTensorEmbedding) and isinstance(obj2, HypergraphTensorEmbedding):
            return self.hypergraph_resonance(obj1, obj2)
        
        if isinstance(obj1, GadgetConstellation) and isinstance(obj2, GadgetConstellation):
            return self.constellation_resonance(obj1, obj2)
        
        # Cross-type resonances
        return self.cross_topology_resonance(obj1, obj2)
    
    def find_resonances(self, 
                       objects: List[Any],
                       min_threshold: Optional[float] = None) -> List[Tuple[int, int, float]]:
        """Find all pairs of objects with significant resonance.
        
        Args:
            objects: List of topological objects
            min_threshold: Minimum resonance threshold (uses self.threshold if None)
            
        Returns:
            List of (index1, index2, resonance_score) tuples
        """
        threshold = min_threshold if min_threshold is not None else self.threshold
        resonances = []
        
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                score = self.compute_resonance(objects[i], objects[j])
                if score >= threshold:
                    resonances.append((i, j, score))
        
        # Sort by resonance strength
        resonances.sort(key=lambda x: x[2], reverse=True)
        
        return resonances
    
    def is_resonant(self, obj1: Any, obj2: Any, threshold: Optional[float] = None) -> bool:
        """Check if two objects are in resonance.
        
        Args:
            obj1: First object
            obj2: Second object
            threshold: Optional custom threshold
            
        Returns:
            True if objects resonate
        """
        threshold = threshold if threshold is not None else self.threshold
        return self.compute_resonance(obj1, obj2) >= threshold


def topological_resonance(structures: List[Any], 
                         threshold: float = 0.7) -> Dict[str, Any]:
    """Perform topological resonance computation on a collection of structures.
    
    This is the main entry point for computing topological matchings in operadic space.
    
    Args:
        structures: List of topological structures (membranes, hypergraphs, constellations)
        threshold: Resonance threshold
        
    Returns:
        Dictionary with resonance analysis results
    """
    matcher = TopologicalMatcher(threshold)
    
    # Find all resonances
    resonances = matcher.find_resonances(structures)
    
    # Build resonance graph
    resonance_graph: Dict[int, List[int]] = {i: [] for i in range(len(structures))}
    for i, j, score in resonances:
        resonance_graph[i].append(j)
        resonance_graph[j].append(i)
    
    # Find connected components (resonance clusters)
    visited = set()
    clusters = []
    
    def dfs(node: int, cluster: List[int]):
        if node in visited:
            return
        visited.add(node)
        cluster.append(node)
        for neighbor in resonance_graph[node]:
            dfs(neighbor, cluster)
    
    for i in range(len(structures)):
        if i not in visited:
            cluster = []
            dfs(i, cluster)
            if len(cluster) > 1:
                clusters.append(cluster)
    
    return {
        "num_structures": len(structures),
        "num_resonances": len(resonances),
        "resonance_pairs": resonances,
        "resonance_clusters": clusters,
        "avg_resonance": np.mean([score for _, _, score in resonances]) if resonances else 0.0,
        "max_resonance": max([score for _, _, score in resonances], default=0.0),
        "topology_types": {type(s).__name__ for s in structures},
    }
