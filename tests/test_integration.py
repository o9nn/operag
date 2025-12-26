"""Integration tests for the full operag system."""

import pytest
import numpy as np
from operag import (
    PrimeFactorMembrane,
    HypergraphTensorEmbedding,
    OperadGadget,
    GadgetConstellation,
    TopologicalMatcher,
    topological_resonance,
)


class TestOperagIntegration:
    """Integration tests for the complete system."""
    
    def test_full_pipeline(self):
        """Test a complete computational pipeline."""
        
        # 1. Create membranes for data compartmentalization
        data_membrane = PrimeFactorMembrane(12, "data")
        code_membrane = PrimeFactorMembrane(18, "code")
        
        # Store data with permissions
        data_membrane.store("input", [1, 2, 3, 4, 5], permission="read")
        code_membrane.store("algorithm", "topological_sort", permission="execute")
        
        # 2. Create hypergraph for relational data
        relations = HypergraphTensorEmbedding(num_nodes=5, embedding_dim=8)
        relations.add_hyperedge((0, 1, 2), weight=1.0)
        relations.add_hyperedge((2, 3, 4), weight=0.8)
        relations.update_embeddings(iterations=5)
        
        # 3. Create computational constellation
        constellation = GadgetConstellation("processor")
        
        preprocess = OperadGadget("preprocess", arity=1, 
                                  transform=lambda x: np.array(x) * 2)
        aggregate = OperadGadget("aggregate", arity=2,
                                transform=lambda x, y: x + y)
        postprocess = OperadGadget("postprocess", arity=1,
                                  transform=lambda x: x / 2)
        
        constellation.add_gadget(preprocess)
        constellation.add_gadget(aggregate)
        constellation.add_gadget(postprocess)
        
        # 4. Perform topological resonance analysis
        structures = [data_membrane, code_membrane, relations]
        results = topological_resonance(structures, threshold=0.3)
        
        # Verify the system works
        assert results["num_structures"] == 3
        assert data_membrane.retrieve("input", permission="read") is not None
        assert relations.num_nodes == 5
        assert len(constellation.gadgets) == 3
    
    def test_membrane_and_hypergraph_alignment(self):
        """Test alignment between membrane topology and hypergraph structure."""
        
        # Create membrane with specific topology
        m = PrimeFactorMembrane(30)  # 2 × 3 × 5, depth=3
        
        # Create hypergraph with related structure
        hg = HypergraphTensorEmbedding(num_nodes=3, embedding_dim=16)
        hg.add_hyperedge((0, 1, 2))
        
        # Check topological resonance
        matcher = TopologicalMatcher(threshold=0.5)
        resonance = matcher.compute_resonance(m, hg)
        
        # They should have some resonance due to structural similarity
        assert 0.0 <= resonance <= 1.0
    
    def test_gadget_composition_with_membranes(self):
        """Test composing gadgets with membrane-stored data."""
        
        # Create membrane with computational data
        m = PrimeFactorMembrane(6)
        m.store("x", 5)
        m.store("y", 3)
        
        # Create gadgets for computation
        add = OperadGadget("add", arity=2, transform=lambda x, y: x + y)
        mul = OperadGadget("mul", arity=2, transform=lambda x, y: x * y)
        
        # Compose gadgets
        composed = add.compose([mul], positions=[0])
        
        # Retrieve data and compute
        x = m.retrieve("x")
        y = m.retrieve("y")
        result = composed(x, y, 10)  # mul(5,3) + 10 = 15 + 10 = 25
        
        assert result == 25
    
    def test_resonance_driven_merging(self):
        """Test merging structures based on topological resonance."""
        
        m1 = PrimeFactorMembrane(6)   # 2 × 3
        m2 = PrimeFactorMembrane(10)  # 2 × 5
        m3 = PrimeFactorMembrane(15)  # 3 × 5
        
        m1.store("data1", "A")
        m2.store("data2", "B")
        m3.store("data3", "C")
        
        # Find which membranes resonate most strongly
        matcher = TopologicalMatcher()
        
        r12 = matcher.compute_resonance(m1, m2)
        r13 = matcher.compute_resonance(m1, m3)
        r23 = matcher.compute_resonance(m2, m3)
        
        # All share at least one factor, so all can merge
        assert r12 > 0
        assert r13 > 0
        assert r23 > 0
        
        # Merge the highest resonance pair
        if r12 >= max(r13, r23):
            merged = m1.merge(m2)
        elif r13 >= r23:
            merged = m1.merge(m3)
        else:
            merged = m2.merge(m3)
        
        # Verify merged membrane has combined data
        assert merged.size > m1.size
        assert len(merged.contents) >= 2
    
    def test_multi_level_topology(self):
        """Test computation across multiple topological levels."""
        
        # Level 1: Membrane topology
        membranes = [
            PrimeFactorMembrane(4),
            PrimeFactorMembrane(8),
            PrimeFactorMembrane(16),
        ]
        
        # Level 2: Hypergraph connections
        hg = HypergraphTensorEmbedding(num_nodes=len(membranes), embedding_dim=8)
        hg.add_hyperedge((0, 1), weight=1.0)
        hg.add_hyperedge((1, 2), weight=1.0)
        
        # Level 3: Operad operations
        constellation = GadgetConstellation("system")
        for i, m in enumerate(membranes):
            g = OperadGadget(f"process_{i}", arity=1,
                           transform=lambda x: x * m.size)
            constellation.add_gadget(g)
        
        # Perform resonance analysis across all levels
        all_structures = membranes + [hg, constellation]
        results = topological_resonance(all_structures, threshold=0.2)
        
        assert results["num_structures"] == 5
        assert results["num_resonances"] >= 1  # Should find some resonances
