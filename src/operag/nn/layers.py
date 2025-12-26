"""Shape and identity layers as structural gadgets.

This module implements shape transformers and structural operations
as graph morphisms with well-typed paths.
"""

import numpy as np
from typing import Tuple, Optional
from .base import NeuralModule, TypeSignature, ShapeConstraint, register_module


@register_module
class Identity(NeuralModule):
    """Identity layer - reflexive operad node.
    
    Passes input through unchanged. Useful for:
    - Debugging and introspection
    - Identity proofs in operad composition
    - Skip connections
    
    Properties:
    - 1-arity operad
    - Linear
    - Shape-preserving
    """
    
    def __init__(self, name: str = "identity"):
        """Initialize Identity layer.
        
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
            shape_constraint=ShapeConstraint(),  # No constraints
            behavior_traits={
                "differentiable": True,
                "non_linear": False,
                "stochastic": False,
                "stateful": False,
            },
            transform=lambda x: x
        )
        
        self.operad_contract["meta"] = {
            "reflexive": True,
            "shape_preserving": True
        }


@register_module
class Reshape(NeuralModule):
    """Reshape layer - shape transformer gadget.
    
    Reshapes tensor to target shape while preserving total size.
    Enforces shape contracts using declarative constraints.
    
    Properties:
    - 1-arity operad
    - Linear
    - Shape-altering (rank-changing)
    """
    
    def __init__(self, target_shape: Tuple[int, ...], name: str = "reshape"):
        """Initialize Reshape layer.
        
        Args:
            target_shape: Target shape (can contain -1 for automatic dimension)
            name: Name of the module
        """
        self.target_shape = target_shape
        
        def reshape_fn(x):
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            return np.reshape(x, self.target_shape)
        
        super().__init__(
            name=name,
            arity=1,
            type_signature=TypeSignature(
                input_type="Tensor",
                output_type="Tensor",
                output_shape=target_shape,
                dtype="float32"
            ),
            shape_constraint=ShapeConstraint(
                # Don't validate size for dynamic shapes with -1
                constraints=["total_size preserved"] if -1 not in target_shape else None
            ),
            behavior_traits={
                "differentiable": True,
                "non_linear": False,
                "stochastic": False,
                "stateful": False,
            },
            transform=reshape_fn
        )
        
        self.operad_contract["meta"] = {
            "shape_transformer": True,
            "target_shape": target_shape,
            "rank_altering": True
        }
    
    def __repr__(self) -> str:
        return f"Reshape({self.name}, target_shape={self.target_shape})"


@register_module
class Flatten(NeuralModule):
    """Flatten layer - converts multi-dimensional tensor to 1D or 2D.
    
    Flattens input while optionally preserving the first dimension (batch).
    
    Properties:
    - 1-arity operad
    - Linear
    - Rank-reducing
    """
    
    def __init__(self, start_dim: int = 1, name: str = "flatten"):
        """Initialize Flatten layer.
        
        Args:
            start_dim: Dimension from which to start flattening (default: 1 to preserve batch)
            name: Name of the module
        """
        self.start_dim = start_dim
        
        def flatten_fn(x):
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            
            if self.start_dim == 0:
                return x.flatten()
            else:
                # Preserve dimensions before start_dim
                new_shape = x.shape[:self.start_dim] + (-1,)
                return x.reshape(new_shape)
        
        super().__init__(
            name=name,
            arity=1,
            type_signature=TypeSignature(
                input_type="Tensor",
                output_type="Tensor",
                dtype="float32"
            ),
            shape_constraint=ShapeConstraint(
                min_rank=1 if start_dim == 0 else 2
            ),
            behavior_traits={
                "differentiable": True,
                "non_linear": False,
                "stochastic": False,
                "stateful": False,
            },
            transform=flatten_fn
        )
        
        self.operad_contract["meta"] = {
            "shape_transformer": True,
            "rank_reducing": True,
            "start_dim": start_dim
        }
    
    def __repr__(self) -> str:
        return f"Flatten({self.name}, start_dim={self.start_dim})"


@register_module
class Mean(NeuralModule):
    """Mean reduction layer - axis-reducing operation.
    
    Computes mean along specified axis.
    
    Properties:
    - 1-arity operad
    - Linear
    - Axis-reducing
    """
    
    def __init__(self, axis: Optional[int] = None, keepdims: bool = False, name: str = "mean"):
        """Initialize Mean layer.
        
        Args:
            axis: Axis along which to compute mean (None = all axes)
            keepdims: Whether to keep reduced dimensions
            name: Name of the module
        """
        self.axis = axis
        self.keepdims = keepdims
        
        def mean_fn(x):
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            return np.mean(x, axis=self.axis, keepdims=self.keepdims)
        
        super().__init__(
            name=name,
            arity=1,
            type_signature=TypeSignature(
                input_type="Tensor",
                output_type="Tensor" if keepdims or axis is not None else "Scalar",
                dtype="float32"
            ),
            shape_constraint=ShapeConstraint(
                min_rank=1
            ),
            behavior_traits={
                "differentiable": True,
                "non_linear": False,
                "stochastic": False,
                "stateful": False,
            },
            transform=mean_fn
        )
        
        self.operad_contract["meta"] = {
            "axis_reducing": True,
            "axis": axis,
            "keepdims": keepdims,
            "aggregation": "mean"
        }
    
    def __repr__(self) -> str:
        return f"Mean({self.name}, axis={self.axis}, keepdims={self.keepdims})"


@register_module
class Max(NeuralModule):
    """Max reduction layer - axis-reducing operation.
    
    Computes maximum along specified axis.
    
    Properties:
    - 1-arity operad
    - Non-linear
    - Axis-reducing
    """
    
    def __init__(self, axis: Optional[int] = None, keepdims: bool = False, name: str = "max"):
        """Initialize Max layer.
        
        Args:
            axis: Axis along which to compute max (None = all axes)
            keepdims: Whether to keep reduced dimensions
            name: Name of the module
        """
        self.axis = axis
        self.keepdims = keepdims
        
        def max_fn(x):
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            return np.max(x, axis=self.axis, keepdims=self.keepdims)
        
        super().__init__(
            name=name,
            arity=1,
            type_signature=TypeSignature(
                input_type="Tensor",
                output_type="Tensor" if keepdims or axis is not None else "Scalar",
                dtype="float32"
            ),
            shape_constraint=ShapeConstraint(
                min_rank=1
            ),
            behavior_traits={
                "differentiable": False,  # Not differentiable everywhere
                "non_linear": True,
                "stochastic": False,
                "stateful": False,
            },
            transform=max_fn
        )
        
        self.operad_contract["meta"] = {
            "axis_reducing": True,
            "axis": axis,
            "keepdims": keepdims,
            "aggregation": "max"
        }
    
    def __repr__(self) -> str:
        return f"Max({self.name}, axis={self.axis}, keepdims={self.keepdims})"


@register_module
class Sum(NeuralModule):
    """Sum reduction layer - axis-reducing operation.
    
    Computes sum along specified axis.
    
    Properties:
    - 1-arity operad
    - Linear
    - Axis-reducing
    """
    
    def __init__(self, axis: Optional[int] = None, keepdims: bool = False, name: str = "sum"):
        """Initialize Sum layer.
        
        Args:
            axis: Axis along which to compute sum (None = all axes)
            keepdims: Whether to keep reduced dimensions
            name: Name of the module
        """
        self.axis = axis
        self.keepdims = keepdims
        
        def sum_fn(x):
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            return np.sum(x, axis=self.axis, keepdims=self.keepdims)
        
        super().__init__(
            name=name,
            arity=1,
            type_signature=TypeSignature(
                input_type="Tensor",
                output_type="Tensor" if keepdims or axis is not None else "Scalar",
                dtype="float32"
            ),
            shape_constraint=ShapeConstraint(
                min_rank=1
            ),
            behavior_traits={
                "differentiable": True,
                "non_linear": False,
                "stochastic": False,
                "stateful": False,
            },
            transform=sum_fn
        )
        
        self.operad_contract["meta"] = {
            "axis_reducing": True,
            "axis": axis,
            "keepdims": keepdims,
            "aggregation": "sum"
        }
    
    def __repr__(self) -> str:
        return f"Sum({self.name}, axis={self.axis}, keepdims={self.keepdims})"
