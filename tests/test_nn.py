"""Tests for neural network operad gadgets."""

import pytest
import numpy as np
from operag.nn import (
    # Base
    NeuralModule, TypeSignature, ShapeConstraint,
    # Activations
    ReLU, Tanh, Sigmoid, Softmax, LeakyReLU,
    # Loss functions
    MSELoss, CrossEntropyLoss, BCELoss, L1Loss,
    # Layers
    Identity, Reshape, Flatten, Mean, Max, Sum,
    # Containers
    Sequential, Parallel, Add, Multiply, Concat, Dot,
)


class TestTypeSignature:
    """Test TypeSignature class."""
    
    def test_initialization(self):
        sig = TypeSignature(
            input_type="Tensor",
            output_type="Tensor",
            input_shape=(10, 20),
            output_shape=(10, 20),
            dtype="float32"
        )
        assert sig.input_type == "Tensor"
        assert sig.output_type == "Tensor"
        assert sig.input_shape == (10, 20)
        assert sig.output_shape == (10, 20)
    
    def test_compatibility(self):
        sig1 = TypeSignature("Tensor", "Tensor", dtype="float32")
        sig2 = TypeSignature("Tensor", "Tensor", dtype="float32")
        assert sig1.is_compatible_with(sig2)
        
        # sig3 outputs Scalar, sig4 expects Scalar input - compatible
        sig3 = TypeSignature("Tensor", "Scalar", dtype="float32")
        sig4 = TypeSignature("Scalar", "Tensor", dtype="float32")
        assert sig3.is_compatible_with(sig4)
        
        # sig1 outputs Tensor, sig4 expects Scalar - incompatible
        assert not sig1.is_compatible_with(sig4)


class TestShapeConstraint:
    """Test ShapeConstraint class."""
    
    def test_min_rank(self):
        constraint = ShapeConstraint(min_rank=2)
        assert constraint.validate((10, 20))
        assert constraint.validate((10, 20, 30))
        assert not constraint.validate((10,))
    
    def test_max_rank(self):
        constraint = ShapeConstraint(max_rank=2)
        assert constraint.validate((10,))
        assert constraint.validate((10, 20))
        assert not constraint.validate((10, 20, 30))
    
    def test_fixed_dims(self):
        constraint = ShapeConstraint(fixed_dims={0: 10, 1: 20})
        assert constraint.validate((10, 20))
        assert constraint.validate((10, 20, 30))
        assert not constraint.validate((5, 20))


class TestActivations:
    """Test activation function modules."""
    
    def test_relu(self):
        relu = ReLU()
        x = np.array([-2, -1, 0, 1, 2])
        result = relu(x)
        expected = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_equal(result, expected)
        
        assert relu.behavior_traits["non_linear"] is True
        assert relu.behavior_traits["differentiable"] is True
    
    def test_tanh(self):
        tanh = Tanh()
        x = np.array([0, 1, -1])
        result = tanh(x)
        expected = np.tanh(x)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_sigmoid(self):
        sigmoid = Sigmoid()
        x = np.array([0, 1, -1])
        result = sigmoid(x)
        
        # Check output range
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        
        # Check sigmoid(0) ≈ 0.5
        assert np.isclose(sigmoid(np.array([0]))[0], 0.5, atol=0.01)
    
    def test_softmax(self):
        softmax = Softmax(axis=-1)
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = softmax(x)
        
        # Check that each row sums to 1
        row_sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0])
        
        # Check all values in [0, 1]
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_leaky_relu(self):
        leaky_relu = LeakyReLU(alpha=0.1)
        x = np.array([-2, -1, 0, 1, 2])
        result = leaky_relu(x)
        expected = np.array([-0.2, -0.1, 0, 1, 2])
        np.testing.assert_array_almost_equal(result, expected)


class TestLossFunctions:
    """Test loss function modules."""
    
    def test_mse_loss(self):
        mse = MSELoss()
        prediction = np.array([1, 2, 3])
        target = np.array([1, 2, 3])
        loss = mse(prediction, target)
        assert np.isclose(loss, 0.0)
        
        prediction = np.array([1, 2, 3])
        target = np.array([2, 3, 4])
        loss = mse(prediction, target)
        assert np.isclose(loss, 1.0)
        
        # Check terminal operad contract
        assert mse.operad_contract["terminal"] is True
        assert mse.type_signature.output_type == "Scalar"
    
    def test_cross_entropy_loss(self):
        ce = CrossEntropyLoss()
        # Perfect prediction
        prediction = np.array([[0.1, 0.9], [0.2, 0.8]])
        target = np.array([[0, 1], [0, 1]])
        loss = ce(prediction, target)
        assert loss > 0  # Should be small but positive
        
        # Check error signal tracking
        assert len(ce.get_error_signals()) > 0
    
    def test_bce_loss(self):
        bce = BCELoss()
        prediction = np.array([0.5, 0.8, 0.2])
        target = np.array([1, 1, 0])
        loss = bce(prediction, target)
        assert loss > 0
        
        # Check metadata
        assert bce.operad_contract["loss_type"] == "binary_classification"
    
    def test_l1_loss(self):
        l1 = L1Loss()
        prediction = np.array([1, 2, 3])
        target = np.array([2, 3, 4])
        loss = l1(prediction, target)
        assert np.isclose(loss, 1.0)


class TestLayers:
    """Test layer modules."""
    
    def test_identity(self):
        identity = Identity()
        x = np.array([1, 2, 3, 4, 5])
        result = identity(x)
        np.testing.assert_array_equal(result, x)
        
        # Check reflexive property
        assert identity.operad_contract["meta"]["reflexive"] is True
    
    def test_reshape(self):
        reshape = Reshape(target_shape=(2, 3))
        x = np.array([1, 2, 3, 4, 5, 6])
        result = reshape(x)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result.flatten(), x)
    
    def test_flatten(self):
        flatten = Flatten(start_dim=1)
        x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Shape: (2, 2, 2)
        result = flatten(x)
        assert result.shape == (2, 4)
        
        # Test flatten all
        flatten_all = Flatten(start_dim=0)
        result_all = flatten_all(x)
        assert result_all.shape == (8,)
    
    def test_mean(self):
        mean = Mean(axis=1)
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = mean(x)
        expected = np.array([2, 5])
        np.testing.assert_array_equal(result, expected)
        
        # Test global mean
        mean_all = Mean(axis=None)
        result_all = mean_all(x)
        assert np.isclose(result_all, 3.5)
    
    def test_max(self):
        max_layer = Max(axis=1)
        x = np.array([[1, 3, 2], [6, 4, 5]])
        result = max_layer(x)
        expected = np.array([3, 6])
        np.testing.assert_array_equal(result, expected)
    
    def test_sum(self):
        sum_layer = Sum(axis=0)
        x = np.array([[1, 2], [3, 4]])
        result = sum_layer(x)
        expected = np.array([4, 6])
        np.testing.assert_array_equal(result, expected)


class TestContainers:
    """Test container modules."""
    
    def test_sequential(self):
        # Create a simple sequential network
        seq = Sequential(
            ReLU("relu1"),
            Tanh("tanh1"),
            name="seq1"
        )
        
        x = np.array([-1, 0, 1, 2])
        result = seq(x)
        
        # Should apply ReLU then Tanh
        intermediate = np.maximum(0, x)  # ReLU
        expected = np.tanh(intermediate)  # Tanh
        np.testing.assert_array_almost_equal(result, expected)
        
        # Check behavior traits
        assert seq.behavior_traits["differentiable"] is True
        assert seq.behavior_traits["non_linear"] is True
    
    def test_sequential_type_checking(self):
        # This should work - compatible types
        seq = Sequential(
            ReLU(),
            Sigmoid()
        )
        assert seq is not None
        
        # Test with incompatible arity - should still work for 1-arity modules
        seq2 = Sequential(
            Identity(),
            ReLU(),
            Tanh()
        )
        x = np.array([1, 2, 3])
        result = seq2(x)
        assert result.shape == x.shape
    
    def test_parallel(self):
        parallel = Parallel(
            ReLU("relu1"),
            Tanh("tanh1"),
            name="parallel1"
        )
        
        x = np.array([1, 2, 3])
        result = parallel(x)
        
        # Should return tuple of results
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        # Check each result
        np.testing.assert_array_equal(result[0], np.maximum(0, x))
        np.testing.assert_array_almost_equal(result[1], np.tanh(x))
    
    def test_add(self):
        add = Add()
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        result = add(x, y)
        expected = np.array([5, 7, 9])
        np.testing.assert_array_equal(result, expected)
        
        # Test broadcasting
        x = np.array([[1, 2], [3, 4]])
        y = np.array([10, 20])
        result = add(x, y)
        expected = np.array([[11, 22], [13, 24]])
        np.testing.assert_array_equal(result, expected)
    
    def test_multiply(self):
        mul = Multiply()
        x = np.array([1, 2, 3])
        y = np.array([2, 3, 4])
        result = mul(x, y)
        expected = np.array([2, 6, 12])
        np.testing.assert_array_equal(result, expected)
    
    def test_concat(self):
        concat = Concat(axis=0)
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        result = concat(x, y)
        expected = np.array([1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(result, expected)
        
        # Test 2D concatenation
        concat2 = Concat(axis=1)
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])
        result = concat2(x, y)
        expected = np.array([[1, 2, 5, 6], [3, 4, 7, 8]])
        np.testing.assert_array_equal(result, expected)
    
    def test_dot(self):
        dot = Dot()
        
        # Vector dot product
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        result = dot(x, y)
        expected = 32  # 1*4 + 2*5 + 3*6
        assert result == expected
        
        # Matrix multiplication
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])
        result = dot(x, y)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(result, expected)


class TestNeuralNetworkComposition:
    """Test composing neural networks with operads."""
    
    def test_simple_mlp(self):
        """Test a simple multi-layer perceptron composition."""
        # Create a simple network: Input -> ReLU -> Sigmoid
        network = Sequential(
            ReLU("layer1"),
            Sigmoid("layer2"),
            name="mlp"
        )
        
        x = np.array([-2, -1, 0, 1, 2])
        result = network(x)
        
        # Check output is in [0, 1] (sigmoid output)
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_residual_connection(self):
        """Test residual connection using Add and Identity."""
        # Create residual block: x + f(x)
        identity_path = Identity("identity")
        transform_path = Tanh("transform")
        
        x = np.array([0.5, 1.0, 1.5])
        
        # Manually compute residual
        identity_out = identity_path(x)
        transform_out = transform_path(x)
        
        add = Add("residual_add")
        result = add(identity_out, transform_out)
        
        expected = x + np.tanh(x)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_classifier_with_loss(self):
        """Test classifier composed with loss function."""
        # Simple classifier: Softmax -> CrossEntropyLoss
        softmax = Softmax(axis=-1, name="classifier")
        loss_fn = CrossEntropyLoss("loss")
        
        # Logits
        logits = np.array([[1.0, 2.0, 0.5], [0.5, 1.5, 2.0]])
        predictions = softmax(logits)
        
        # One-hot targets
        targets = np.array([[0, 1, 0], [0, 0, 1]])
        
        loss = loss_fn(predictions, targets)
        assert loss > 0
        assert isinstance(loss, (float, np.floating))
    
    def test_metadata_propagation(self):
        """Test that metadata is properly propagated through composition."""
        network = Sequential(
            ReLU("relu"),
            Tanh("tanh"),
            name="network"
        )
        
        metadata = network.get_metadata()
        
        assert metadata["name"] == "network"
        assert metadata["arity"] == 1
        assert metadata["behavior_traits"]["differentiable"] is True
        assert metadata["behavior_traits"]["non_linear"] is True
        assert metadata["topology_type"] == "sequential"


class TestModuleRegistry:
    """Test module registration system."""
    
    def test_modules_registered(self):
        from operag.nn.base import get_registered_modules
        
        registry = get_registered_modules()
        
        # Check that key modules are registered
        assert "ReLU" in registry
        assert "Tanh" in registry
        assert "Sequential" in registry
        assert "MSELoss" in registry
        assert "Identity" in registry
        
        # Check we can instantiate from registry
        relu_class = registry["ReLU"]
        relu = relu_class()
        assert isinstance(relu, NeuralModule)
