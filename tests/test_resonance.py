"""Tests for Topological Resonance."""

import pytest
import numpy as np
from operag.membranes import PrimeFactorMembrane
from operag.hypergraph import HypergraphTensorEmbedding
from operag.operad import OperadGadget, GadgetConstellation
from operag.resonance import TopologicalMatcher, topological_resonance


class TestTopologicalMatcher:
    """Tests for TopologicalMatcher class."""
    
    def test_initialization(self):
        matcher = TopologicalMatcher(threshold=0.7)
        assert matcher.threshold == 0.7
    
    def test_membrane_resonance_identical(self):
        matcher = TopologicalMatcher()
        m1 = PrimeFactorMembrane(12)
        m2 = PrimeFactorMembrane(12)
        
        resonance = matcher.membrane_resonance(m1, m2)
        assert resonance == 1.0  # Identical topologies
    
    def test_membrane_resonance_similar(self):
        matcher = TopologicalMatcher()
        m1 = PrimeFactorMembrane(12)  # 2 × 2 × 3
        m2 = PrimeFactorMembrane(18)  # 2 × 3 × 3
        
        resonance = matcher.membrane_resonance(m1, m2)
        assert resonance > 0.0  # Share factors 2 and 3
        # Note: Due to set-based Jaccard similarity, [2,2,3] and [2,3,3] 
        # both become {2,3}, so they have perfect resonance
    
    def test_membrane_resonance_different(self):
        matcher = TopologicalMatcher()
        m1 = PrimeFactorMembrane(7)   # Prime
        m2 = PrimeFactorMembrane(11)  # Different prime
        
        resonance = matcher.membrane_resonance(m1, m2)
        assert resonance == 0.0  # No common factors
    
    def test_hypergraph_resonance(self):
        matcher = TopologicalMatcher()
        
        hg1 = HypergraphTensorEmbedding(num_nodes=10, embedding_dim=16)
        hg1.add_hyperedge((0, 1, 2))
        hg1.add_hyperedge((2, 3, 4))
        
        hg2 = HypergraphTensorEmbedding(num_nodes=10, embedding_dim=16)
        hg2.add_hyperedge((5, 6, 7))
        hg2.add_hyperedge((7, 8, 9))
        
        resonance = matcher.hypergraph_resonance(hg1, hg2)
        assert 0.0 <= resonance <= 1.0
        assert resonance > 0.5  # Similar structure
    
    def test_constellation_resonance(self):
        matcher = TopologicalMatcher()
        
        c1 = GadgetConstellation("c1")
        g1 = OperadGadget("f", arity=2, topology_type="sequential")
        c1.add_gadget(g1)
        
        c2 = GadgetConstellation("c2")
        g2 = OperadGadget("g", arity=2, topology_type="sequential")
        c2.add_gadget(g2)
        
        resonance = matcher.constellation_resonance(c1, c2)
        assert resonance > 0.7  # Very similar
    
    def test_cross_topology_resonance_membrane_hypergraph(self):
        matcher = TopologicalMatcher()
        
        m = PrimeFactorMembrane(12)  # depth = 3
        hg = HypergraphTensorEmbedding(num_nodes=3, embedding_dim=16)
        
        resonance = matcher.cross_topology_resonance(m, hg)
        assert 0.0 <= resonance <= 1.0
    
    def test_compute_resonance_same_type(self):
        matcher = TopologicalMatcher()
        
        m1 = PrimeFactorMembrane(12)
        m2 = PrimeFactorMembrane(12)
        
        resonance = matcher.compute_resonance(m1, m2)
        assert resonance == 1.0
    
    def test_compute_resonance_cross_type(self):
        matcher = TopologicalMatcher()
        
        m = PrimeFactorMembrane(12)
        hg = HypergraphTensorEmbedding(num_nodes=10, embedding_dim=16)
        
        resonance = matcher.compute_resonance(m, hg)
        assert 0.0 <= resonance <= 1.0
    
    def test_is_resonant(self):
        matcher = TopologicalMatcher(threshold=0.5)
        
        m1 = PrimeFactorMembrane(12)  # 2 × 2 × 3
        m2 = PrimeFactorMembrane(18)  # 2 × 3 × 3
        
        assert matcher.is_resonant(m1, m2)  # Should have high resonance
    
    def test_find_resonances(self):
        matcher = TopologicalMatcher(threshold=0.5)
        
        structures = [
            PrimeFactorMembrane(12),  # 2 × 2 × 3
            PrimeFactorMembrane(18),  # 2 × 3 × 3
            PrimeFactorMembrane(7),   # Prime
        ]
        
        resonances = matcher.find_resonances(structures)
        
        assert len(resonances) >= 1  # At least m1 and m2 should resonate
        # Check format
        for i, j, score in resonances:
            assert 0 <= i < len(structures)
            assert 0 <= j < len(structures)
            assert i < j
            assert 0.0 <= score <= 1.0


class TestTopologicalResonanceFunction:
    """Tests for topological_resonance function."""
    
    def test_basic_resonance(self):
        structures = [
            PrimeFactorMembrane(12),
            PrimeFactorMembrane(18),
            PrimeFactorMembrane(6),
        ]
        
        results = topological_resonance(structures, threshold=0.3)
        
        assert results["num_structures"] == 3
        assert results["num_resonances"] >= 1
        assert "resonance_pairs" in results
        assert "resonance_clusters" in results
        assert "avg_resonance" in results
    
    def test_mixed_types_resonance(self):
        m = PrimeFactorMembrane(12)
        
        hg = HypergraphTensorEmbedding(num_nodes=10, embedding_dim=16)
        hg.add_hyperedge((0, 1, 2))
        
        c = GadgetConstellation()
        g = OperadGadget("f", arity=2)
        c.add_gadget(g)
        
        structures = [m, hg, c]
        results = topological_resonance(structures, threshold=0.1)
        
        assert results["num_structures"] == 3
        assert len(results["topology_types"]) == 3  # All different types
    
    def test_no_resonances(self):
        structures = [
            PrimeFactorMembrane(7),
            PrimeFactorMembrane(11),
            PrimeFactorMembrane(13),
        ]
        
        # High threshold, different primes - no resonances
        results = topological_resonance(structures, threshold=0.9)
        
        assert results["num_resonances"] == 0
        assert len(results["resonance_clusters"]) == 0
        assert results["avg_resonance"] == 0.0
    
    def test_resonance_clusters(self):
        structures = [
            PrimeFactorMembrane(12),  # 2 × 2 × 3
            PrimeFactorMembrane(18),  # 2 × 3 × 3 - should cluster with 12
            PrimeFactorMembrane(7),   # Prime - isolated
            PrimeFactorMembrane(6),   # 2 × 3 - should cluster with 12, 18
        ]
        
        results = topological_resonance(structures, threshold=0.3)
        
        # Should find at least one cluster with indices 0, 1, 3
        clusters = results["resonance_clusters"]
        assert len(clusters) >= 1
