"""Base classes for neural network operad gadgets.

This module provides the categorical core infrastructure for neural network
components as typed operadic agents within a semantic network.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
from ..operad.gadget_constellation import OperadGadget


@dataclass
class TypeSignature:
    """Type signature for neural network modules.
    
    Defines the input/output types and shapes for type-safe composition.
    """
    
    input_type: str  # e.g., "Tensor", "Index", "Scalar"
    output_type: str
    input_shape: Optional[Tuple[int, ...]] = None  # Can be None for dynamic shapes
    output_shape: Optional[Tuple[int, ...]] = None
    dtype: str = "float32"
    
    def is_compatible_with(self, other: 'TypeSignature') -> bool:
        """Check if this signature is compatible with another."""
        # Output must match input type
        if self.output_type != other.input_type:
            return False
        
        # If shapes are specified, they must match
        if self.output_shape is not None and other.input_shape is not None:
            if self.output_shape != other.input_shape:
                return False
        
        # Data types should be compatible
        if self.dtype != other.dtype:
            return False
        
        return True
    
    def __repr__(self) -> str:
        shape_str = f"{self.input_shape} → {self.output_shape}" if self.input_shape else "dynamic"
        return f"TypeSignature({self.input_type} → {self.output_type}, {shape_str}, {self.dtype})"


@dataclass
class ShapeConstraint:
    """Shape constraints for neural network operations."""
    
    min_rank: Optional[int] = None  # Minimum tensor rank
    max_rank: Optional[int] = None  # Maximum tensor rank
    fixed_dims: Optional[Dict[int, int]] = None  # Fixed dimensions at specific positions
    constraints: Optional[List[str]] = None  # Custom constraint expressions
    
    def validate(self, shape: Tuple[int, ...]) -> bool:
        """Validate a shape against constraints."""
        rank = len(shape)
        
        if self.min_rank is not None and rank < self.min_rank:
            return False
        
        if self.max_rank is not None and rank > self.max_rank:
            return False
        
        if self.fixed_dims is not None:
            for pos, expected_dim in self.fixed_dims.items():
                if pos >= rank or shape[pos] != expected_dim:
                    return False
        
        return True
    
    def __repr__(self) -> str:
        parts = []
        if self.min_rank is not None:
            parts.append(f"rank≥{self.min_rank}")
        if self.max_rank is not None:
            parts.append(f"rank≤{self.max_rank}")
        if self.fixed_dims:
            parts.append(f"dims={self.fixed_dims}")
        return f"ShapeConstraint({', '.join(parts)})" if parts else "ShapeConstraint(none)"


class NeuralModule(OperadGadget):
    """Base class for neural network modules as typed operadic gadgets.
    
    Extends OperadGadget with neural network-specific metadata:
    - Type signature for input/output types
    - Shape constraints for tensor operations
    - Behavior traits (differentiable, non-linear, stochastic, etc.)
    """
    
    def __init__(
        self,
        name: str,
        arity: int = 1,
        type_signature: Optional[TypeSignature] = None,
        shape_constraint: Optional[ShapeConstraint] = None,
        behavior_traits: Optional[Dict[str, bool]] = None,
        **kwargs
    ):
        """Initialize a neural module.
        
        Args:
            name: Name of the module
            arity: Number of inputs (default: 1 for most neural layers)
            type_signature: Type signature for the module
            shape_constraint: Shape constraints for validation
            behavior_traits: Behavior metadata (differentiable, non_linear, etc.)
            **kwargs: Additional arguments passed to OperadGadget
        """
        super().__init__(name=name, arity=arity, **kwargs)
        
        # Default type signature if not provided
        self.type_signature = type_signature or TypeSignature(
            input_type="Tensor",
            output_type="Tensor",
            dtype="float32"
        )
        
        self.shape_constraint = shape_constraint or ShapeConstraint()
        
        # Default behavior traits
        self.behavior_traits = behavior_traits or {
            "differentiable": True,
            "non_linear": False,
            "stochastic": False,
            "stateful": False,
        }
        
        # Operad contract (arity and type constraints)
        self.operad_contract = {
            "arity": arity,
            "input_types": [self.type_signature.input_type] * arity,
            "output_type": self.type_signature.output_type,
            "constraints": []
        }
    
    def validate_input(self, *inputs) -> bool:
        """Validate inputs against type signature and shape constraints.
        
        Args:
            *inputs: Input tensors/values
            
        Returns:
            True if inputs are valid
        """
        if len(inputs) != self.arity:
            return False
        
        for inp in inputs:
            # Check if input is numpy array (tensor)
            if isinstance(inp, np.ndarray):
                if not self.shape_constraint.validate(inp.shape):
                    return False
        
        return True
    
    def forward(self, *inputs):
        """Forward pass of the module.
        
        This should be overridden by subclasses. By default, uses the transform.
        
        Args:
            *inputs: Input tensors
            
        Returns:
            Output tensor
        """
        if not self.validate_input(*inputs):
            raise ValueError(f"Invalid inputs for {self.name}")
        
        return self.transform(*inputs)
    
    def __call__(self, *inputs):
        """Make module callable via forward pass."""
        return self.forward(*inputs)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get module metadata including type and behavior information."""
        return {
            "name": self.name,
            "arity": self.arity,
            "type_signature": str(self.type_signature),
            "shape_constraint": str(self.shape_constraint),
            "behavior_traits": self.behavior_traits,
            "operad_contract": self.operad_contract,
            "topology_type": self.topology_type,
        }
    
    def can_compose_with(self, other: 'NeuralModule', position: int = 0) -> bool:
        """Check if this module can compose with another.
        
        Enhanced with type signature checking.
        
        Args:
            other: Another neural module
            position: Position for composition
            
        Returns:
            True if composition is valid
        """
        # First check base operad compatibility
        if not super().can_compose_with(other, position):
            return False
        
        # Check type signature compatibility if both have them
        if hasattr(other, 'type_signature') and self.type_signature:
            if not other.type_signature.is_compatible_with(self.type_signature):
                return False
        
        return True
    
    def __repr__(self) -> str:
        traits_str = ", ".join(k for k, v in self.behavior_traits.items() if v)
        return f"{self.__class__.__name__}({self.name}, arity={self.arity}, traits=[{traits_str}])"


# Type registry for neural network modules
_MODULE_REGISTRY: Dict[str, type] = {}


def register_module(module_class: type) -> type:
    """Register a neural module class in the type registry.
    
    Args:
        module_class: Module class to register
        
    Returns:
        The same class (decorator pattern)
    """
    _MODULE_REGISTRY[module_class.__name__] = module_class
    return module_class


def get_registered_modules() -> Dict[str, type]:
    """Get all registered neural module classes."""
    return _MODULE_REGISTRY.copy()


def resolve_type_signature(module_name: str) -> Optional[TypeSignature]:
    """Resolve type signature for a registered module.
    
    Args:
        module_name: Name of the module class
        
    Returns:
        Type signature if module is registered, None otherwise
    """
    if module_name not in _MODULE_REGISTRY:
        return None
    
    module_class = _MODULE_REGISTRY[module_name]
    
    # Try to get a default instance to extract signature
    try:
        instance = module_class()
        return instance.type_signature
    except:
        return None
