"""Hypergraph Tensor Embeddings.

This module implements tensor-based embeddings for hypergraphs, allowing
representation of multi-way relationships as higher-order tensors in a
topological space.
"""

from typing import List, Dict, Set, Tuple, Optional, Any
import numpy as np
from scipy.sparse import csr_matrix


class HypergraphTensorEmbedding:
    """A hypergraph represented as tensor embeddings in topological space.
    
    In this model:
    - Nodes are embedded as vectors in a latent space
    - Hyperedges are represented as tensors capturing multi-way relationships
    - The topology emerges from the tensor structure
    """
    
    def __init__(self, num_nodes: int, embedding_dim: int, order: int = 3):
        """Initialize a hypergraph tensor embedding.
        
        Args:
            num_nodes: Number of nodes in the hypergraph
            embedding_dim: Dimensionality of the embedding space
            order: Order of the tensor (2=graph, 3=3-way hypergraph, etc.)
        """
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.order = order
        
        # Initialize node embeddings randomly
        self.node_embeddings = np.random.randn(num_nodes, embedding_dim) * 0.1
        
        # Hyperedges stored as lists of node indices
        self.hyperedges: List[Tuple[int, ...]] = []
        
        # Edge weights
        self.edge_weights: Dict[Tuple[int, ...], float] = {}
        
        # Tensor representation (constructed on demand)
        self._tensor_cache: Optional[np.ndarray] = None
        self._tensor_dirty = True
        
    def add_hyperedge(self, nodes: Tuple[int, ...], weight: float = 1.0):
        """Add a hyperedge connecting multiple nodes.
        
        Args:
            nodes: Tuple of node indices forming the hyperedge
            weight: Weight/strength of the hyperedge
        """
        if len(nodes) > self.order:
            raise ValueError(f"Hyperedge order {len(nodes)} exceeds tensor order {self.order}")
        
        # Pad with sentinel value if needed
        padded = nodes + (-1,) * (self.order - len(nodes))
        
        self.hyperedges.append(padded)
        self.edge_weights[padded] = weight
        self._tensor_dirty = True
        
    def get_tensor_representation(self) -> np.ndarray:
        """Get the tensor representation of the hypergraph.
        
        Returns:
            Tensor of shape (num_nodes,) * order representing the hypergraph
        """
        if not self._tensor_dirty and self._tensor_cache is not None:
            return self._tensor_cache
        
        # Build tensor
        shape = (self.num_nodes,) * self.order
        tensor = np.zeros(shape)
        
        for edge in self.hyperedges:
            # Skip edges with padding
            valid_indices = [i for i in edge if i >= 0]
            if len(valid_indices) == len(edge):
                tensor[edge] = self.edge_weights[edge]
        
        self._tensor_cache = tensor
        self._tensor_dirty = False
        return tensor
    
    def compute_node_degree_tensor(self) -> np.ndarray:
        """Compute degree tensor for each node.
        
        Returns:
            Array of shape (num_nodes,) with degree information
        """
        degrees = np.zeros(self.num_nodes)
        
        for edge in self.hyperedges:
            for node in edge:
                if node >= 0:
                    degrees[node] += self.edge_weights.get(edge, 1.0)
        
        return degrees
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Project hypergraph to a 2D adjacency matrix.
        
        This creates a graph projection by connecting all pairs of nodes
        that appear together in hyperedges.
        
        Returns:
            Adjacency matrix of shape (num_nodes, num_nodes)
        """
        adj = np.zeros((self.num_nodes, self.num_nodes))
        
        for edge in self.hyperedges:
            valid_nodes = [i for i in edge if i >= 0]
            weight = self.edge_weights.get(edge, 1.0)
            
            # Connect all pairs in the hyperedge
            for i, node_i in enumerate(valid_nodes):
                for node_j in valid_nodes[i+1:]:
                    adj[node_i, node_j] += weight
                    adj[node_j, node_i] += weight
        
        return adj
    
    def embed_hyperedge(self, nodes: Tuple[int, ...]) -> np.ndarray:
        """Compute embedding vector for a hyperedge.
        
        Args:
            nodes: Tuple of node indices
            
        Returns:
            Embedding vector for the hyperedge
        """
        valid_nodes = [i for i in nodes if i >= 0 and i < self.num_nodes]
        if not valid_nodes:
            return np.zeros(self.embedding_dim)
        
        # Use mean pooling of node embeddings
        embeddings = self.node_embeddings[valid_nodes]
        return np.mean(embeddings, axis=0)
    
    def topological_distance(self, edge1: Tuple[int, ...], edge2: Tuple[int, ...]) -> float:
        """Compute topological distance between two hyperedges.
        
        Args:
            edge1: First hyperedge
            edge2: Second hyperedge
            
        Returns:
            Distance in embedding space
        """
        emb1 = self.embed_hyperedge(edge1)
        emb2 = self.embed_hyperedge(edge2)
        return float(np.linalg.norm(emb1 - emb2))
    
    def get_topology_signature(self) -> Dict[str, Any]:
        """Get topological signature of the hypergraph.
        
        Returns:
            Dictionary with topological properties
        """
        tensor = self.get_tensor_representation()
        degrees = self.compute_node_degree_tensor()
        
        return {
            "num_nodes": self.num_nodes,
            "num_edges": len(self.hyperedges),
            "embedding_dim": self.embedding_dim,
            "order": self.order,
            "max_degree": float(np.max(degrees)),
            "avg_degree": float(np.mean(degrees)),
            "tensor_norm": float(np.linalg.norm(tensor)),
        }
    
    def update_embeddings(self, learning_rate: float = 0.01, iterations: int = 10):
        """Update node embeddings to better capture hypergraph structure.
        
        Uses a simple gradient-based update to minimize reconstruction error.
        
        Args:
            learning_rate: Step size for updates
            iterations: Number of update iterations
        """
        for _ in range(iterations):
            for edge in self.hyperedges:
                valid_nodes = [i for i in edge if i >= 0]
                if len(valid_nodes) < 2:
                    continue
                
                # Compute mean embedding
                mean_emb = np.mean(self.node_embeddings[valid_nodes], axis=0)
                
                # Update each node to be closer to the mean
                for node in valid_nodes:
                    diff = mean_emb - self.node_embeddings[node]
                    self.node_embeddings[node] += learning_rate * diff
        
        self._tensor_dirty = True
    
    def __repr__(self) -> str:
        return (f"HypergraphTensorEmbedding(nodes={self.num_nodes}, "
                f"edges={len(self.hyperedges)}, dim={self.embedding_dim}, order={self.order})")
