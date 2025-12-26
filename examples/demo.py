#!/usr/bin/env python3
"""
Example demonstrating the Operag topological computing framework.

This example shows how computation emerges from topological resonance
between membranes, hypergraphs, and operad gadgets.
"""

import numpy as np
from operag import (
    PrimeFactorMembrane,
    HypergraphTensorEmbedding,
    OperadGadget,
    GadgetConstellation,
    TopologicalMatcher,
    topological_resonance,
)


def main():
    print("=" * 70)
    print("OPERAG: Topological Operad-driven Cognitive Computing")
    print("Computation as Topological Resonance")
    print("=" * 70)
    print()
    
    # =========================================================================
    # PART 1: Prime-Factor Shaped Membrane Topologies
    # =========================================================================
    print("1. PRIME-FACTOR SHAPED MEMBRANE TOPOLOGIES")
    print("-" * 70)
    
    # Create membranes with different topologies
    m1 = PrimeFactorMembrane(12, "DataStore")    # 2 × 2 × 3
    m2 = PrimeFactorMembrane(18, "CodeVault")    # 2 × 3 × 3
    m3 = PrimeFactorMembrane(30, "PermissionGate")  # 2 × 3 × 5
    
    print(f"Created membranes:")
    print(f"  {m1}")
    print(f"  {m2}")
    print(f"  {m3}")
    print()
    
    # Store data with permission gating
    m1.store("user_data", [1, 2, 3, 4, 5], permission="read")
    m2.store("algorithm", "topological_sort", permission="execute")
    m3.store("access_key", "secret123", permission="admin")
    
    print("Data stored with permission gating:")
    print(f"  DataStore['user_data'] = {m1.retrieve('user_data', permission='read')}")
    print(f"  CodeVault['algorithm'] = {m2.retrieve('algorithm', permission='execute')}")
    print(f"  PermissionGate['access_key'] (no permission) = {m3.retrieve('access_key')}")
    print(f"  PermissionGate['access_key'] (with admin) = {m3.retrieve('access_key', permission='admin')}")
    print()
    
    # Merge compatible membranes
    if m1.can_merge_with(m2):
        merged = m1.merge(m2)
        print(f"Merged membranes: {merged}")
        print(f"  Contains: {list(merged.contents.keys())}")
    print()
    
    # =========================================================================
    # PART 2: Hypergraph Tensor Embeddings
    # =========================================================================
    print("2. HYPERGRAPH TENSOR EMBEDDINGS")
    print("-" * 70)
    
    # Create a hypergraph representing relationships
    hg = HypergraphTensorEmbedding(num_nodes=10, embedding_dim=16, order=3)
    
    # Add multi-way relationships
    hg.add_hyperedge((0, 1, 2), weight=1.0)
    hg.add_hyperedge((2, 3, 4), weight=0.9)
    hg.add_hyperedge((4, 5, 6), weight=0.8)
    hg.add_hyperedge((1, 5, 8), weight=0.7)
    
    print(f"Created hypergraph: {hg}")
    
    # Update embeddings to capture topology
    hg.update_embeddings(learning_rate=0.01, iterations=10)
    
    sig = hg.get_topology_signature()
    print(f"\nTopological signature:")
    print(f"  Nodes: {sig['num_nodes']}")
    print(f"  Hyperedges: {sig['num_edges']}")
    print(f"  Avg degree: {sig['avg_degree']:.2f}")
    print(f"  Tensor norm: {sig['tensor_norm']:.2f}")
    
    # Compute distances in topological space
    dist = hg.topological_distance((0, 1, 2), (4, 5, 6))
    print(f"\nTopological distance between edges (0,1,2) and (4,5,6): {dist:.4f}")
    print()
    
    # =========================================================================
    # PART 3: Operad Gadget Constellations
    # =========================================================================
    print("3. OPERAD GADGET CONSTELLATIONS")
    print("-" * 70)
    
    # Create computational gadgets
    add = OperadGadget("add", arity=2, transform=lambda x, y: x + y)
    mul = OperadGadget("mul", arity=2, transform=lambda x, y: x * y)
    scale = OperadGadget("scale", arity=1, transform=lambda x: x * 2)
    square = OperadGadget("square", arity=1, transform=lambda x: x ** 2)
    
    print("Created gadgets: add, mul, scale, square")
    print()
    
    # Compose gadgets operadically
    # Create: add(mul(x, y), scale(z))
    composed = add.compose([mul, scale], positions=[0, 1])
    
    result = composed(3, 4, 5)
    print(f"Composed computation: add(mul(3,4), scale(5))")
    print(f"  = add(12, 10) = {result}")
    print()
    
    # Build a constellation
    constellation = GadgetConstellation("DataProcessor")
    constellation.add_gadget(square)
    constellation.add_gadget(scale)
    constellation.add_gadget(add)
    constellation.add_gadget(mul)
    
    # Connect gadgets
    constellation.connect("square", "scale", position=0)
    constellation.connect("scale", "add", position=0)
    constellation.connect("mul", "add", position=1)
    
    print("Constellation topology:")
    print(constellation.visualize_topology())
    print()
    
    # =========================================================================
    # PART 4: Topological Resonance - Computation as Pattern Matching
    # =========================================================================
    print("4. TOPOLOGICAL RESONANCE")
    print("-" * 70)
    
    # Collect all topological structures
    structures = [m1, m2, m3, hg, constellation]
    
    print(f"Computing resonances between {len(structures)} structures...")
    print()
    
    # Compute pairwise resonances
    matcher = TopologicalMatcher(threshold=0.5)
    
    print("Pairwise resonances:")
    print(f"  DataStore ↔ CodeVault: {matcher.compute_resonance(m1, m2):.3f}")
    print(f"  DataStore ↔ PermissionGate: {matcher.compute_resonance(m1, m3):.3f}")
    print(f"  CodeVault ↔ PermissionGate: {matcher.compute_resonance(m2, m3):.3f}")
    print(f"  DataStore ↔ Hypergraph: {matcher.compute_resonance(m1, hg):.3f}")
    print(f"  Hypergraph ↔ Constellation: {matcher.compute_resonance(hg, constellation):.3f}")
    print()
    
    # Perform full resonance analysis
    results = topological_resonance(structures, threshold=0.3)
    
    print("Global resonance analysis:")
    print(f"  Total structures: {results['num_structures']}")
    print(f"  Resonant pairs: {results['num_resonances']}")
    print(f"  Average resonance: {results['avg_resonance']:.3f}")
    print(f"  Max resonance: {results['max_resonance']:.3f}")
    print(f"  Topology types: {results['topology_types']}")
    
    if results['resonance_clusters']:
        print(f"\n  Resonance clusters found:")
        for i, cluster in enumerate(results['resonance_clusters']):
            print(f"    Cluster {i+1}: {cluster}")
    
    print()
    print("=" * 70)
    print("Computation complete. Topological patterns have resonated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
