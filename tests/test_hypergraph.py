"""Tests for Hypergraph Tensor Embeddings."""

import pytest
import numpy as np
from operag.hypergraph import HypergraphTensorEmbedding


class TestHypergraphTensorEmbedding:
    """Tests for HypergraphTensorEmbedding class."""
    
    def test_initialization(self):
        hg = HypergraphTensorEmbedding(num_nodes=10, embedding_dim=16, order=3)
        assert hg.num_nodes == 10
        assert hg.embedding_dim == 16
        assert hg.order == 3
        assert hg.node_embeddings.shape == (10, 16)
    
    def test_add_hyperedge(self):
        hg = HypergraphTensorEmbedding(num_nodes=5, embedding_dim=8, order=3)
        hg.add_hyperedge((0, 1, 2), weight=1.0)
        
        assert len(hg.hyperedges) == 1
        assert (0, 1, 2) in hg.edge_weights
    
    def test_add_hyperedge_with_padding(self):
        hg = HypergraphTensorEmbedding(num_nodes=5, embedding_dim=8, order=3)
        hg.add_hyperedge((0, 1), weight=0.5)
        
        assert len(hg.hyperedges) == 1
        # Should be padded with -1
        assert hg.hyperedges[0] == (0, 1, -1)
    
    def test_tensor_representation(self):
        hg = HypergraphTensorEmbedding(num_nodes=5, embedding_dim=8, order=3)
        hg.add_hyperedge((0, 1, 2), weight=1.0)
        
        tensor = hg.get_tensor_representation()
        assert tensor.shape == (5, 5, 5)
        assert tensor[0, 1, 2] == 1.0
    
    def test_compute_node_degree(self):
        hg = HypergraphTensorEmbedding(num_nodes=5, embedding_dim=8, order=3)
        hg.add_hyperedge((0, 1, 2), weight=1.0)
        hg.add_hyperedge((0, 2, 3), weight=1.0)
        
        degrees = hg.compute_node_degree_tensor()
        assert degrees[0] == 2.0  # Node 0 in 2 edges
        assert degrees[1] == 1.0  # Node 1 in 1 edge
        assert degrees[2] == 2.0  # Node 2 in 2 edges
        assert degrees[4] == 0.0  # Node 4 in 0 edges
    
    def test_adjacency_matrix(self):
        hg = HypergraphTensorEmbedding(num_nodes=5, embedding_dim=8, order=3)
        hg.add_hyperedge((0, 1, 2), weight=1.0)
        
        adj = hg.get_adjacency_matrix()
        assert adj.shape == (5, 5)
        
        # All pairs in hyperedge should be connected
        assert adj[0, 1] == 1.0
        assert adj[0, 2] == 1.0
        assert adj[1, 2] == 1.0
        
        # Symmetric
        assert adj[1, 0] == 1.0
    
    def test_embed_hyperedge(self):
        hg = HypergraphTensorEmbedding(num_nodes=5, embedding_dim=8, order=3)
        emb = hg.embed_hyperedge((0, 1, 2))
        
        assert emb.shape == (8,)
        # Should be mean of node embeddings
        expected = np.mean(hg.node_embeddings[[0, 1, 2]], axis=0)
        np.testing.assert_array_almost_equal(emb, expected)
    
    def test_topological_distance(self):
        hg = HypergraphTensorEmbedding(num_nodes=5, embedding_dim=8, order=3)
        
        dist = hg.topological_distance((0, 1, 2), (0, 1, 2))
        assert dist == 0.0  # Same edge, zero distance
        
        dist = hg.topological_distance((0, 1, 2), (3, 4))
        assert dist > 0.0  # Different edges, positive distance
    
    def test_topology_signature(self):
        hg = HypergraphTensorEmbedding(num_nodes=10, embedding_dim=16, order=3)
        hg.add_hyperedge((0, 1, 2))
        hg.add_hyperedge((2, 3, 4))
        
        sig = hg.get_topology_signature()
        assert sig["num_nodes"] == 10
        assert sig["num_edges"] == 2
        assert sig["embedding_dim"] == 16
        assert sig["order"] == 3
    
    def test_update_embeddings(self):
        hg = HypergraphTensorEmbedding(num_nodes=5, embedding_dim=8, order=3)
        hg.add_hyperedge((0, 1, 2))
        
        old_embeddings = hg.node_embeddings.copy()
        hg.update_embeddings(learning_rate=0.01, iterations=5)
        
        # Embeddings should change
        assert not np.allclose(hg.node_embeddings, old_embeddings)
