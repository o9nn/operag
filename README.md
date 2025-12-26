# Operag

**A nonlinear, topology-gated, operad-driven cognitive computing model** — like an **Antikythera machine**, a **P system**, a **biomimetic operating system** where **data, code, permissions, and computation are all *topological phenomena*.**

## Overview

Operag implements a novel computational paradigm based on **topological matchings** in operadic space, where computation emerges from the resonance of topological structures.

### Core Components

#### 1. Prime-Factor Shaped Membrane Topologies

P-system inspired computational membranes whose structure is determined by prime factorization. Each membrane creates nested compartments based on prime factors, implementing hierarchical computation with topologically-constrained data flow and permission boundaries.

```python
from operag import PrimeFactorMembrane

# Create a membrane with topology shaped by 12 = 2 × 2 × 3
membrane = PrimeFactorMembrane(size=12, label="M12")
print(membrane)  # M12: 12 = 2 × 2 × 3 (depth=3)

# Store data with permission gating
membrane.store("data", [1, 2, 3], permission="read")

# Membranes can merge if topologically compatible
m1 = PrimeFactorMembrane(6)  # 2 × 3
m2 = PrimeFactorMembrane(10)  # 2 × 5
if m1.can_merge_with(m2):
    merged = m1.merge(m2)  # Creates membrane with size 60
```

#### 2. Hypergraph Tensor Embeddings

Multi-relational data representation as higher-order tensors in topological space. Nodes are embedded as vectors, and hyperedges are represented as tensors capturing multi-way relationships.

```python
from operag import HypergraphTensorEmbedding

# Create hypergraph with 10 nodes, 16-dim embeddings, order-3 tensors
hg = HypergraphTensorEmbedding(num_nodes=10, embedding_dim=16, order=3)

# Add hyperedges (multi-way relationships)
hg.add_hyperedge((0, 1, 2), weight=1.0)
hg.add_hyperedge((2, 3, 4), weight=0.8)

# Update embeddings to capture structure
hg.update_embeddings(learning_rate=0.01, iterations=10)

# Get topological signature
sig = hg.get_topology_signature()
print(f"Hypergraph: {sig['num_nodes']} nodes, {sig['num_edges']} edges")
```

#### 3. Operad Gadget Constellations

Composable computational structures using operadic composition. Gadgets are topological units of computation that can be composed to form complex computational constellations.

```python
from operag import OperadGadget, GadgetConstellation
import numpy as np

# Create computational gadgets
add_gadget = OperadGadget("add", arity=2, transform=lambda x, y: x + y)
mul_gadget = OperadGadget("mul", arity=2, transform=lambda x, y: x * y)
scale_gadget = OperadGadget("scale", arity=1, transform=lambda x: x * 2)

# Compose gadgets operadically
# (x, y) → add(mul(x, y), scale(x))
composed = add_gadget.compose([mul_gadget, scale_gadget], positions=[0, 1])
result = composed(3, 4)  # mul(3,4) + scale(3) = 12 + 6 = 18

# Build a constellation
constellation = GadgetConstellation("example")
constellation.add_gadget(add_gadget)
constellation.add_gadget(mul_gadget)
constellation.connect("mul", "add", position=0)

print(constellation.visualize_topology())
```

#### 3.5. Neural Network Operad Gadgets

Neural network components as typed operadic agents with type signatures, shape constraints, and behavior traits. Build type-safe, composable neural architectures.

```python
from operag.nn import (
    # Activations
    ReLU, Tanh, Sigmoid, Softmax,
    # Loss functions
    MSELoss, CrossEntropyLoss,
    # Layers
    Identity, Reshape, Flatten,
    # Containers
    Sequential, Parallel, Add
)
import numpy as np

# Build a simple neural network as operadic composition
network = Sequential(
    ReLU("layer1"),
    Tanh("layer2"),
    name="mlp"
)

# Forward pass with type checking
x = np.array([1.0, 2.0, 3.0])
output = network(x)

# Residual connection using operad gadgets
identity = Identity("skip")
transform = Tanh("transform")
add_layer = Add("residual")

residual = add_layer(identity(x), transform(x))  # x + tanh(x)

# Loss function as terminal operad
mse = MSELoss("mse")
predictions = np.array([1.0, 2.0, 3.0])
targets = np.array([1.1, 2.0, 2.9])
loss = mse(predictions, targets)

# Check type signatures
print(f"Network type: {network.type_signature}")
print(f"Behavior: differentiable={network.behavior_traits['differentiable']}")
```

#### 4. Topological Resonance — Computation as Pattern Matching

The core concept: **computation == topological resonance**. Compatible structures resonate and interact, performing computation through topological pattern matching.

```python
from operag import TopologicalMatcher, topological_resonance
from operag import PrimeFactorMembrane, HypergraphTensorEmbedding

# Create various topological structures
m1 = PrimeFactorMembrane(12)  # 2 × 2 × 3
m2 = PrimeFactorMembrane(18)  # 2 × 3 × 3
m3 = PrimeFactorMembrane(15)  # 3 × 5

hg1 = HypergraphTensorEmbedding(num_nodes=10, embedding_dim=16)
hg1.add_hyperedge((0, 1, 2))
hg1.add_hyperedge((2, 3, 4))

# Compute topological resonances
matcher = TopologicalMatcher(threshold=0.5)

# Check if structures resonate
print(f"m1 ↔ m2 resonance: {matcher.compute_resonance(m1, m2):.3f}")
print(f"m1 ↔ m3 resonance: {matcher.compute_resonance(m1, m3):.3f}")

# Find all resonances in a collection
structures = [m1, m2, m3, hg1]
results = topological_resonance(structures, threshold=0.3)

print(f"\nFound {results['num_resonances']} resonances")
print(f"Average resonance: {results['avg_resonance']:.3f}")
print(f"Resonance clusters: {results['resonance_clusters']}")
```

## Installation

```bash
# Clone the repository
git clone https://github.com/o9nn/operag.git
cd operag

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/
```

## Conceptual Foundation

Operag is inspired by:

- **Operads** from category theory — mathematical structures for composing operations
- **P systems** — membrane computing with hierarchical compartments
- **Topological computing** — where structure determines computation
- **The Antikythera mechanism** — ancient analog computer demonstrating mechanical computation through geometric/topological relationships

### Philosophy

In Operag, computation is not sequential instruction execution but rather the **resonance of topological structures**:

- **Data** has topological shape (membrane structure, hypergraph topology)
- **Code** is operadic composition (gadget constellations)
- **Permissions** are membrane boundaries
- **Computation** emerges from topological pattern matching and resonance

This creates a biomimetic computing model where the topology itself is the computation.

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.