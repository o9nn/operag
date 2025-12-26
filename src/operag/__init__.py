"""
Operag - Topological Operad-driven Cognitive Computing

A nonlinear, topology-gated, operad-driven cognitive computing model where
data, code, permissions, and computation are all topological phenomena.

Key Components:
- Prime-Factor Shaped Membrane Topologies: P-system inspired computational membranes
- Hypergraph Tensor Embeddings: Multi-relational data representation
- Operad Gadget Constellations: Composable computational structures
- Topological Resonance: Computation through topological pattern matching
"""

__version__ = "0.1.0"

from .membranes.prime_membranes import PrimeFactorMembrane
from .hypergraph.tensor_embeddings import HypergraphTensorEmbedding
from .operad.gadget_constellation import OperadGadget, GadgetConstellation
from .resonance.topological_matcher import TopologicalMatcher, topological_resonance

__all__ = [
    "PrimeFactorMembrane",
    "HypergraphTensorEmbedding",
    "OperadGadget",
    "GadgetConstellation",
    "TopologicalMatcher",
    "topological_resonance",
]
