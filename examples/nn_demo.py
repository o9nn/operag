#!/usr/bin/env python3
"""
Example demonstrating Neural Network Operad Gadgets.

This example shows how to build neural networks as typed operadic compositions,
with type safety, shape constraints, and behavior traits.
"""

import numpy as np
from operag.nn import (
    # Activations
    ReLU, Tanh, Sigmoid, Softmax,
    # Loss functions
    MSELoss, CrossEntropyLoss, BCELoss,
    # Layers
    Identity, Reshape, Flatten, Mean,
    # Containers
    Sequential, Parallel, Add, Multiply, Concat,
)
from operag.nn.base import get_registered_modules


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    print_section("OPERAG: Neural Network Operad Gadgets")
    print("Type-safe, composable neural network components\n")
    
    # =========================================================================
    # PART 1: Basic Activation Functions as 1-Arity Operads
    # =========================================================================
    print_section("1. ACTIVATION FUNCTIONS AS OPERADIC PRIMITIVES")
    
    # Create activation functions
    relu = ReLU("relu")
    tanh = Tanh("tanh")
    sigmoid = Sigmoid("sigmoid")
    
    # Test input
    x = np.array([-2, -1, 0, 1, 2])
    
    print(f"\nInput: {x}")
    print(f"ReLU(x):    {relu(x)}")
    print(f"Tanh(x):    {tanh(x)}")
    print(f"Sigmoid(x): {sigmoid(x)}")
    
    # Show metadata
    print(f"\nReLU metadata:")
    print(f"  Type signature: {relu.type_signature}")
    print(f"  Behavior traits: {relu.behavior_traits}")
    print(f"  Operad contract: {relu.operad_contract['meta']}")
    
    # =========================================================================
    # PART 2: Sequential Composition - Building a Simple MLP
    # =========================================================================
    print_section("2. SEQUENTIAL COMPOSITION - SIMPLE MLP")
    
    # Build a 2-layer network: Input -> ReLU -> Tanh
    mlp = Sequential(
        ReLU("layer1"),
        Tanh("layer2"),
        name="simple_mlp"
    )
    
    print(f"\nNetwork: {mlp}")
    print(f"Topology: {mlp.topology_type}")
    print(f"Differentiable: {mlp.behavior_traits['differentiable']}")
    print(f"Non-linear: {mlp.behavior_traits['non_linear']}")
    
    # Forward pass
    x = np.array([-1, 0, 1, 2, 3])
    output = mlp(x)
    print(f"\nInput:  {x}")
    print(f"Output: {output}")
    
    # =========================================================================
    # PART 3: Parallel Computation Paths
    # =========================================================================
    print_section("3. PARALLEL COMPUTATION PATHS")
    
    # Create parallel branches
    parallel = Parallel(
        ReLU("branch1"),
        Sigmoid("branch2"),
        Tanh("branch3"),
        name="parallel_branches"
    )
    
    x = np.array([0.5, 1.0, 1.5])
    results = parallel(x)
    
    print(f"\nInput: {x}")
    print(f"Branch 1 (ReLU):    {results[0]}")
    print(f"Branch 2 (Sigmoid): {results[1]}")
    print(f"Branch 3 (Tanh):    {results[2]}")
    
    # =========================================================================
    # PART 4: Residual Connections
    # =========================================================================
    print_section("4. RESIDUAL CONNECTIONS")
    
    # Build residual block manually: x + tanh(x)
    identity = Identity("skip")
    transform = Tanh("transform")
    add = Add("residual_add")
    
    x = np.array([0.5, 1.0, 1.5, 2.0])
    
    identity_out = identity(x)
    transform_out = transform(x)
    residual_out = add(identity_out, transform_out)
    
    print(f"\nInput x:        {x}")
    print(f"Identity(x):    {identity_out}")
    print(f"Tanh(x):        {transform_out}")
    print(f"x + Tanh(x):    {residual_out}")
    
    # =========================================================================
    # PART 5: Shape Transformations
    # =========================================================================
    print_section("5. SHAPE TRANSFORMATIONS")
    
    # Reshape and flatten operations
    x = np.arange(12)
    print(f"\nOriginal shape: {x.shape} -> {x}")
    
    reshape = Reshape(target_shape=(3, 4), name="reshape")
    reshaped = reshape(x)
    print(f"\nReshape to (3,4):")
    print(reshaped)
    
    flatten = Flatten(start_dim=0, name="flatten")
    flattened = flatten(reshaped)
    print(f"\nFlatten: {flattened.shape} -> {flattened}")
    
    # Reduction operations
    x = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"\n2D Input:\n{x}")
    
    mean_layer = Mean(axis=1, name="mean")
    mean_result = mean_layer(x)
    print(f"Mean(axis=1): {mean_result}")
    
    # =========================================================================
    # PART 6: Loss Functions as Terminal Operads
    # =========================================================================
    print_section("6. LOSS FUNCTIONS AS TERMINAL OPERADS")
    
    # MSE Loss for regression
    mse = MSELoss("mse")
    predictions = np.array([1.0, 2.0, 3.0, 4.0])
    targets = np.array([1.1, 2.2, 2.9, 4.1])
    
    loss = mse(predictions, targets)
    print(f"\nMSE Loss:")
    print(f"  Predictions: {predictions}")
    print(f"  Targets:     {targets}")
    print(f"  Loss:        {loss:.6f}")
    print(f"  Terminal operad: {mse.operad_contract['terminal']}")
    
    # Cross-Entropy Loss for classification
    ce = CrossEntropyLoss("cross_entropy")
    
    # Predictions (after softmax)
    predictions = np.array([[0.7, 0.2, 0.1],
                           [0.1, 0.8, 0.1]])
    
    # One-hot encoded targets
    targets = np.array([[1, 0, 0],
                       [0, 1, 0]])
    
    loss = ce(predictions, targets)
    print(f"\nCross-Entropy Loss:")
    print(f"  Predictions:\n{predictions}")
    print(f"  Targets:\n{targets}")
    print(f"  Loss: {loss:.6f}")
    
    # Check error signal tracking
    signals = ce.get_error_signals()
    print(f"  Error signals tracked: {len(signals)}")
    
    # =========================================================================
    # PART 7: Complete Classifier Example
    # =========================================================================
    print_section("7. COMPLETE CLASSIFIER EXAMPLE")
    
    # Build a classifier: Flatten -> ReLU -> Softmax
    classifier = Sequential(
        Flatten(start_dim=0, name="flatten_input"),
        ReLU("hidden_activation"),
        name="classifier"
    )
    
    # Input: 2x3 matrix
    x = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
    
    print(f"\nInput shape: {x.shape}")
    print(f"Input:\n{x}")
    
    # Forward pass through classifier
    hidden = classifier(x)
    print(f"\nAfter classifier: shape={hidden.shape}")
    print(f"Output: {hidden}")
    
    # Apply softmax for class probabilities
    # Note: reshape to 2D for softmax
    logits = hidden.reshape(1, -1)
    softmax = Softmax(axis=-1, name="output_softmax")
    probabilities = softmax(logits)
    
    print(f"\nClass probabilities:")
    print(f"  Shape: {probabilities.shape}")
    print(f"  Probs: {probabilities}")
    print(f"  Sum: {np.sum(probabilities):.6f}")
    
    # =========================================================================
    # PART 8: Type Registry and Introspection
    # =========================================================================
    print_section("8. TYPE REGISTRY AND INTROSPECTION")
    
    registry = get_registered_modules()
    print(f"\nRegistered neural modules: {len(registry)}")
    print("\nModule types:")
    
    activations = [name for name in registry if name in ['ReLU', 'Tanh', 'Sigmoid', 'Softmax', 'LeakyReLU']]
    losses = [name for name in registry if 'Loss' in name]
    layers = [name for name in registry if name in ['Identity', 'Reshape', 'Flatten', 'Mean', 'Max', 'Sum']]
    containers = [name for name in registry if name in ['Sequential', 'Parallel', 'Add', 'Multiply', 'Concat', 'Dot']]
    
    print(f"  Activations: {', '.join(activations)}")
    print(f"  Loss functions: {', '.join(losses)}")
    print(f"  Layers: {', '.join(layers)}")
    print(f"  Containers: {', '.join(containers)}")
    
    # =========================================================================
    # PART 9: Type Signature Validation
    # =========================================================================
    print_section("9. TYPE SIGNATURE VALIDATION")
    
    # Demonstrate type checking
    print("\nType signature compatibility:")
    
    relu = ReLU("relu")
    sigmoid = Sigmoid("sigmoid")
    mse = MSELoss("mse")
    
    print(f"\nReLU type signature:    {relu.type_signature}")
    print(f"Sigmoid type signature: {sigmoid.type_signature}")
    print(f"MSE type signature:     {mse.type_signature}")
    
    # Check compatibility
    relu_sigmoid_compatible = relu.type_signature.is_compatible_with(sigmoid.type_signature)
    sigmoid_mse_compatible = sigmoid.type_signature.is_compatible_with(mse.type_signature)
    
    print(f"\nReLU -> Sigmoid compatible: {relu_sigmoid_compatible}")
    print(f"Sigmoid -> MSE compatible:  {sigmoid_mse_compatible}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_section("SUMMARY")
    print("""
Key Features Demonstrated:

✓ Activation functions as 1-arity operad gadgets
✓ Sequential composition with type checking
✓ Parallel computation paths
✓ Residual connections using Add gadget
✓ Shape transformations (Reshape, Flatten, Mean)
✓ Loss functions as terminal operads with error tracking
✓ Complete classifier pipeline
✓ Type registry and introspection
✓ Type signature validation

Neural networks are now composable operadic gadgets with:
- Type signatures for input/output types
- Shape constraints for validation
- Behavior traits (differentiable, non-linear, etc.)
- Operad contracts for composition rules

This enables topological, shape-aware, behavior-constrained composition!
    """)
    
    print("=" * 70)


if __name__ == "__main__":
    main()
