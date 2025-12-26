# Neural Network Operad Gadgets Implementation

## Overview

This document describes the implementation of neural network components as typed operadic gadgets in the Operag framework. The implementation extends Operag's operad-driven cognitive computing model with type-safe, composable neural network primitives.

## Architecture

### Core Concept

Neural network modules are implemented as **typed operadic agents** that extend the base `OperadGadget` class with:
- **Type signatures**: Input/output type specifications
- **Shape constraints**: Tensor dimension validation
- **Behavior traits**: Metadata about differentiability, non-linearity, etc.
- **Operad contracts**: Composition rules and constraints

### Class Hierarchy

```
OperadGadget (base class from operag.operad)
    ↓
NeuralModule (base class for all neural components)
    ↓
    ├── Activations (ReLU, Tanh, Sigmoid, Softmax, LeakyReLU)
    ├── Loss Functions (MSELoss, CrossEntropyLoss, BCELoss, L1Loss)
    ├── Layers (Identity, Reshape, Flatten, Mean, Max, Sum)
    └── Containers (Sequential, Parallel, Add, Multiply, Concat, Dot)
```

## Module Categories

### 1. Activations (1-arity operads)

**Properties**:
- Single input, single output
- Non-linear (except Identity)
- Differentiable
- Shape-preserving

**Implementations**:
- `ReLU`: max(0, x) - Rectified Linear Unit
- `Tanh`: Hyperbolic tangent
- `Sigmoid`: 1 / (1 + e^-x) - Logistic function
- `Softmax`: Normalized exponential (probability distribution)
- `LeakyReLU`: max(αx, x) - ReLU with small negative slope

**Example**:
```python
from operag.nn import ReLU, Tanh
import numpy as np

relu = ReLU("layer1")
x = np.array([-1, 0, 1, 2])
output = relu(x)  # [0, 0, 1, 2]

# Check metadata
print(relu.behavior_traits)  # {'differentiable': True, 'non_linear': True, ...}
```

### 2. Loss Functions (2-arity terminal operads)

**Properties**:
- Two inputs: prediction and target
- Single scalar output
- Terminal operads (end of computation chain)
- Track error signals

**Implementations**:
- `MSELoss`: Mean Squared Error - regression
- `CrossEntropyLoss`: Cross-Entropy - classification
- `BCELoss`: Binary Cross-Entropy - binary classification
- `L1Loss`: Mean Absolute Error - robust regression

**Example**:
```python
from operag.nn import MSELoss, CrossEntropyLoss
import numpy as np

mse = MSELoss("mse")
predictions = np.array([1.0, 2.0, 3.0])
targets = np.array([1.1, 2.0, 2.9])
loss = mse(predictions, targets)  # 0.0067

# Check error signals
signals = mse.get_error_signals()
print(f"Tracked {len(signals)} error signals")
```

### 3. Layers (shape transformers)

**Properties**:
- Transform tensor shapes
- Preserve or reduce dimensions
- Linear operations

**Implementations**:
- `Identity`: Pass-through (reflexive operad)
- `Reshape`: Change tensor shape
- `Flatten`: Reduce to 1D or 2D
- `Mean`: Reduce along axis (average)
- `Max`: Reduce along axis (maximum)
- `Sum`: Reduce along axis (sum)

**Example**:
```python
from operag.nn import Reshape, Flatten, Mean
import numpy as np

# Reshape
reshape = Reshape(target_shape=(2, 3))
x = np.array([1, 2, 3, 4, 5, 6])
reshaped = reshape(x)  # shape: (2, 3)

# Flatten
flatten = Flatten(start_dim=1)
x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
flattened = flatten(x)  # (2, 4)

# Mean reduction
mean = Mean(axis=1)
x = np.array([[1, 2, 3], [4, 5, 6]])
result = mean(x)  # [2, 5]
```

### 4. Containers (higher-order operads)

**Properties**:
- Compose multiple modules
- Type-checked composition
- Various topologies

**Implementations**:
- `Sequential`: Chain composition f_n ∘ ... ∘ f_1
- `Parallel`: Multiple branches on same input
- `Add`: Element-wise addition
- `Multiply`: Element-wise multiplication
- `Concat`: Concatenation along dimension
- `Dot`: Matrix multiplication

**Example**:
```python
from operag.nn import Sequential, Parallel, Add
from operag.nn import ReLU, Tanh, Identity
import numpy as np

# Sequential composition
mlp = Sequential(
    ReLU("layer1"),
    Tanh("layer2"),
    name="mlp"
)
x = np.array([1, 2, 3])
output = mlp(x)

# Parallel branches
parallel = Parallel(
    ReLU("branch1"),
    Tanh("branch2"),
    name="parallel"
)
results = parallel(x)  # tuple of outputs

# Residual connection
identity = Identity("skip")
transform = Tanh("transform")
add_layer = Add("residual")
residual = add_layer(identity(x), transform(x))  # x + tanh(x)
```

## Type System

### TypeSignature

Defines input/output types and shapes:

```python
from operag.nn.base import TypeSignature

sig = TypeSignature(
    input_type="Tensor",
    output_type="Tensor",
    input_shape=(10, 20),
    output_shape=(10, 20),
    dtype="float32"
)

# Check compatibility
sig1 = TypeSignature("Tensor", "Tensor", dtype="float32")
sig2 = TypeSignature("Tensor", "Scalar", dtype="float32")
compatible = sig1.is_compatible_with(sig2)  # Checks if sig1.output == sig2.input
```

### ShapeConstraint

Validates tensor dimensions:

```python
from operag.nn.base import ShapeConstraint

constraint = ShapeConstraint(
    min_rank=2,           # At least 2D
    max_rank=4,           # At most 4D
    fixed_dims={0: 10}    # First dimension must be 10
)

valid = constraint.validate((10, 20))      # True
invalid = constraint.validate((5, 20))     # False (first dim != 10)
```

### Behavior Traits

Metadata about module properties:

```python
{
    "differentiable": True,   # Can compute gradients
    "non_linear": True,       # Non-linear transformation
    "stochastic": False,      # Deterministic
    "stateful": False         # No internal state
}
```

## Module Registry

All neural modules are registered for runtime introspection:

```python
from operag.nn.base import get_registered_modules

registry = get_registered_modules()
print(f"Registered modules: {len(registry)}")

# Get module class by name
relu_class = registry["ReLU"]
relu = relu_class()
```

## Operad Composition Rules

### Type Compatibility

Modules can compose if output type matches input type:

```python
relu = ReLU()   # Tensor → Tensor
tanh = Tanh()   # Tensor → Tensor
mse = MSELoss() # Tensor → Scalar

# Valid: ReLU → Tanh (both Tensor → Tensor)
seq1 = Sequential(relu, tanh)

# Valid: Tanh → MSE (Tensor output → Tensor input)
# (though MSE needs two inputs, so this is conceptual)
```

### Arity Matching

Container modules must respect input arity:

```python
# Sequential: first module's arity determines container arity
seq = Sequential(
    ReLU("layer1"),  # arity=1
    Tanh("layer2")   # arity=1
)  # Container arity=1

# Add requires arity=2
add = Add()  # arity=2
result = add(x, y)  # Two inputs required
```

## Usage Patterns

### Building a Simple Classifier

```python
from operag.nn import Sequential, ReLU, Softmax, CrossEntropyLoss
import numpy as np

# Build classifier
classifier = Sequential(
    ReLU("hidden"),
    Softmax("output"),
    name="classifier"
)

# Forward pass
logits = np.array([[1.0, 2.0, 0.5], [0.5, 1.5, 2.0]])
probabilities = classifier(logits)

# Compute loss
loss_fn = CrossEntropyLoss()
targets = np.array([[0, 1, 0], [0, 0, 1]])
loss = loss_fn(probabilities, targets)
```

### Residual Block

```python
from operag.nn import Identity, ReLU, Add
import numpy as np

# Create residual block: x + ReLU(x)
identity = Identity("skip")
activation = ReLU("transform")
add_layer = Add("residual")

x = np.array([1.0, 2.0, 3.0])
skip = identity(x)
activated = activation(x)
output = add_layer(skip, activated)  # x + ReLU(x)
```

### Multi-Branch Network

```python
from operag.nn import Parallel, ReLU, Tanh, Sigmoid, Concat
import numpy as np

# Create parallel branches
parallel = Parallel(
    ReLU("branch1"),
    Tanh("branch2"),
    Sigmoid("branch3")
)

# Apply to input
x = np.array([1.0, 2.0, 3.0])
branch_outputs = parallel(x)  # tuple of 3 arrays

# Optionally concatenate
concat = Concat(axis=0)
combined = concat(*branch_outputs)
```

## Testing

Comprehensive test suite with 32 tests covering:
- Type signature validation
- Shape constraint checking
- Activation functions
- Loss functions
- Layer operations
- Container composition
- Neural network pipelines
- Module registry

Run tests:
```bash
pytest tests/test_nn.py -v
```

All 90 tests pass (58 existing + 32 new).

## Examples

### Running the Demo

```bash
python examples/nn_demo.py
```

The demo demonstrates:
1. Activation functions as operadic primitives
2. Sequential composition (MLP)
3. Parallel computation paths
4. Residual connections
5. Shape transformations
6. Loss functions as terminal operads
7. Complete classifier pipeline
8. Type registry and introspection
9. Type signature validation

## Security

- ✅ CodeQL security scan: 0 vulnerabilities
- ✅ No unsafe operations
- ✅ Proper input validation
- ✅ Numerically stable implementations

## Performance Considerations

- Uses NumPy for efficient tensor operations
- Numerically stable implementations (e.g., log-sum-exp in softmax)
- Minimal overhead from type checking
- Shape validation happens at composition time, not runtime

## Future Extensions

Potential additions to the neural network module:

1. **Convolutional layers**: Conv1D, Conv2D, MaxPool
2. **Recurrent layers**: LSTM, GRU cells
3. **Normalization layers**: BatchNorm, LayerNorm
4. **Dropout**: Stochastic regularization
5. **Attention mechanisms**: Self-attention, cross-attention
6. **Graph neural networks**: Message passing on graphs
7. **Optimization**: Gradient descent as operad gadget
8. **Automatic differentiation**: Backward pass tracking

## References

- Original Operag framework: Topological operad-driven computing
- Category theory: Operads and composition
- Neural networks: Standard deep learning components
- Type theory: Type-safe composition rules

## Summary

The Neural Network Operad Gadgets implementation successfully extends Operag with:

- ✅ 21 neural network modules across 4 categories
- ✅ Type signature system for safe composition
- ✅ Shape constraint validation
- ✅ Behavior trait propagation
- ✅ Module registry for introspection
- ✅ Comprehensive testing (32 new tests)
- ✅ Security validated (0 vulnerabilities)
- ✅ Complete documentation and examples

This creates a **topological, shape-aware, behavior-constrained** neural network framework where neural networks are composable operadic structures, consistent with Operag's metagraph design philosophy.
