"""Activation functions as 1-arity operadic primitives.

This module implements neural network activation functions as composable
operad gadgets with type constraints and differentiability metadata.
"""

import numpy as np
from typing import Optional
from .base import NeuralModule, TypeSignature, ShapeConstraint, register_module


@register_module
class ReLU(NeuralModule):
    """Rectified Linear Unit activation function.
    
    Applies the element-wise function: ReLU(x) = max(0, x)
    
    Properties:
    - 1-arity operad (single input)
    - Non-linear
    - Differentiable (almost everywhere)
    - Shape-preserving
    """
    
    def __init__(self, name: str = "relu"):
        """Initialize ReLU activation.
        
        Args:
            name: Name of the module
        """
        super().__init__(
            name=name,
            arity=1,
            type_signature=TypeSignature(
                input_type="Tensor",
                output_type="Tensor",
                dtype="float32"
            ),
            shape_constraint=ShapeConstraint(
                min_rank=1  # At least 1D tensor
            ),
            behavior_traits={
                "differentiable": True,
                "non_linear": True,
                "stochastic": False,
                "stateful": False,
            },
            transform=lambda x: np.maximum(0, x)
        )
        
        # Update operad contract
        self.operad_contract["meta"] = {
            "differentiable": True,
            "non_linear": True,
            "activation": "relu"
        }


@register_module
class Tanh(NeuralModule):
    """Hyperbolic tangent activation function.
    
    Applies the element-wise function: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    
    Properties:
    - 1-arity operad (single input)
    - Non-linear
    - Differentiable
    - Shape-preserving
    - Output range: (-1, 1)
    """
    
    def __init__(self, name: str = "tanh"):
        """Initialize Tanh activation.
        
        Args:
            name: Name of the module
        """
        super().__init__(
            name=name,
            arity=1,
            type_signature=TypeSignature(
                input_type="Tensor",
                output_type="Tensor",
                dtype="float32"
            ),
            shape_constraint=ShapeConstraint(
                min_rank=1
            ),
            behavior_traits={
                "differentiable": True,
                "non_linear": True,
                "stochastic": False,
                "stateful": False,
            },
            transform=lambda x: np.tanh(x)
        )
        
        self.operad_contract["meta"] = {
            "differentiable": True,
            "non_linear": True,
            "activation": "tanh",
            "output_range": (-1, 1)
        }


@register_module
class Sigmoid(NeuralModule):
    """Sigmoid activation function.
    
    Applies the element-wise function: sigmoid(x) = 1 / (1 + e^-x)
    
    Properties:
    - 1-arity operad (single input)
    - Non-linear
    - Differentiable
    - Shape-preserving
    - Output range: (0, 1)
    """
    
    def __init__(self, name: str = "sigmoid"):
        """Initialize Sigmoid activation.
        
        Args:
            name: Name of the module
        """
        def sigmoid_fn(x):
            # Numerically stable sigmoid
            return np.where(
                x >= 0,
                1 / (1 + np.exp(-x)),
                np.exp(x) / (1 + np.exp(x))
            )
        
        super().__init__(
            name=name,
            arity=1,
            type_signature=TypeSignature(
                input_type="Tensor",
                output_type="Tensor",
                dtype="float32"
            ),
            shape_constraint=ShapeConstraint(
                min_rank=1
            ),
            behavior_traits={
                "differentiable": True,
                "non_linear": True,
                "stochastic": False,
                "stateful": False,
            },
            transform=sigmoid_fn
        )
        
        self.operad_contract["meta"] = {
            "differentiable": True,
            "non_linear": True,
            "activation": "sigmoid",
            "output_range": (0, 1)
        }


@register_module
class Softmax(NeuralModule):
    """Softmax activation function.
    
    Applies the softmax function along the specified axis:
    softmax(x)[i] = exp(x[i]) / sum(exp(x[j]))
    
    Properties:
    - 1-arity operad (single input)
    - Non-linear
    - Differentiable
    - Shape-preserving
    - Output: probability distribution (sums to 1)
    - Works on 1D vectors or higher-dimensional tensors
    """
    
    def __init__(self, axis: int = -1, name: str = "softmax"):
        """Initialize Softmax activation.
        
        Args:
            axis: Axis along which to apply softmax
            name: Name of the module
        """
        self.axis = axis
        
        def softmax_fn(x):
            # Numerically stable softmax
            x_shifted = x - np.max(x, axis=self.axis, keepdims=True)
            exp_x = np.exp(x_shifted)
            return exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
        
        super().__init__(
            name=name,
            arity=1,
            type_signature=TypeSignature(
                input_type="Tensor",
                output_type="Tensor",
                dtype="float32"
            ),
            shape_constraint=ShapeConstraint(
                min_rank=1  # Can work on 1D vectors
            ),
            behavior_traits={
                "differentiable": True,
                "non_linear": True,
                "stochastic": False,
                "stateful": False,
            },
            transform=softmax_fn
        )
        
        self.operad_contract["meta"] = {
            "differentiable": True,
            "non_linear": True,
            "activation": "softmax",
            "axis": axis,
            "output_constraint": "probability_distribution"
        }
    
    def __repr__(self) -> str:
        return f"Softmax({self.name}, axis={self.axis})"


@register_module
class LeakyReLU(NeuralModule):
    """Leaky ReLU activation function.
    
    Applies the element-wise function: LeakyReLU(x) = max(alpha*x, x)
    
    Properties:
    - 1-arity operad (single input)
    - Non-linear
    - Differentiable everywhere
    - Shape-preserving
    """
    
    def __init__(self, alpha: float = 0.01, name: str = "leaky_relu"):
        """Initialize Leaky ReLU activation.
        
        Args:
            alpha: Negative slope coefficient
            name: Name of the module
        """
        self.alpha = alpha
        
        super().__init__(
            name=name,
            arity=1,
            type_signature=TypeSignature(
                input_type="Tensor",
                output_type="Tensor",
                dtype="float32"
            ),
            shape_constraint=ShapeConstraint(
                min_rank=1
            ),
            behavior_traits={
                "differentiable": True,
                "non_linear": True,
                "stochastic": False,
                "stateful": False,
            },
            transform=lambda x: np.where(x > 0, x, self.alpha * x)
        )
        
        self.operad_contract["meta"] = {
            "differentiable": True,
            "non_linear": True,
            "activation": "leaky_relu",
            "alpha": alpha
        }
    
    def __repr__(self) -> str:
        return f"LeakyReLU({self.name}, alpha={self.alpha})"
