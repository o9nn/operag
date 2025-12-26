"""Container modules for composing neural network layers.

This module implements higher-order operad gadgets for composing
neural network modules with type validation and shape constraints.
"""

import numpy as np
from typing import List, Optional, Tuple
from .base import NeuralModule, TypeSignature, ShapeConstraint, register_module


@register_module
class Sequential(NeuralModule):
    """Sequential container - operad chain composition.
    
    Composes modules in sequence: f_n ∘ ... ∘ f_2 ∘ f_1
    Enforces strict arity matching and type compatibility.
    
    Properties:
    - Variable arity (depends on first module)
    - Type-checked composition
    - Chain topology
    """
    
    def __init__(self, *modules: NeuralModule, name: str = "sequential"):
        """Initialize Sequential container.
        
        Args:
            *modules: Modules to compose in sequence
            name: Name of the container
        """
        if not modules:
            raise ValueError("Sequential requires at least one module")
        
        self.modules = list(modules)
        
        # Validate type compatibility
        for i in range(len(self.modules) - 1):
            current = self.modules[i]
            next_module = self.modules[i + 1]
            
            if hasattr(current, 'type_signature') and hasattr(next_module, 'type_signature'):
                if not current.type_signature.is_compatible_with(next_module.type_signature):
                    raise ValueError(
                        f"Type mismatch at position {i}: "
                        f"{current.type_signature} not compatible with {next_module.type_signature}"
                    )
        
        # Create composed transform
        def sequential_transform(*inputs):
            # Handle first module
            if len(inputs) == 1:
                result = self.modules[0](inputs[0])
            else:
                result = self.modules[0](*inputs)
            
            # Apply remaining modules
            for module in self.modules[1:]:
                result = module(result)
            
            return result
        
        # Use first module's arity and last module's output type
        first_module = self.modules[0]
        last_module = self.modules[-1]
        
        super().__init__(
            name=name,
            arity=first_module.arity,
            type_signature=TypeSignature(
                input_type=first_module.type_signature.input_type,
                output_type=last_module.type_signature.output_type,
                input_shape=first_module.type_signature.input_shape,
                output_shape=last_module.type_signature.output_shape,
                dtype=first_module.type_signature.dtype
            ),
            shape_constraint=first_module.shape_constraint,
            behavior_traits={
                "differentiable": all(m.behavior_traits.get("differentiable", False) for m in self.modules),
                "non_linear": any(m.behavior_traits.get("non_linear", False) for m in self.modules),
                "stochastic": any(m.behavior_traits.get("stochastic", False) for m in self.modules),
                "stateful": any(m.behavior_traits.get("stateful", False) for m in self.modules),
            },
            transform=sequential_transform,
            topology_type="sequential"
        )
        
        self.operad_contract["meta"] = {
            "container": "sequential",
            "num_modules": len(self.modules),
            "depth": len(self.modules)
        }
    
    def __repr__(self) -> str:
        modules_str = " → ".join(m.name for m in self.modules)
        return f"Sequential({modules_str})"


@register_module
class Parallel(NeuralModule):
    """Parallel container - parallel computation paths.
    
    Applies multiple modules to the same input independently,
    returning a tuple of results.
    
    Properties:
    - 1-arity operad (single input, multiple branches)
    - Parallel topology
    - Outputs tuple
    """
    
    def __init__(self, *modules: NeuralModule, name: str = "parallel"):
        """Initialize Parallel container.
        
        Args:
            *modules: Modules to apply in parallel
            name: Name of the container
        """
        if not modules:
            raise ValueError("Parallel requires at least one module")
        
        self.modules = list(modules)
        
        # All modules should have same input type
        first_input_type = self.modules[0].type_signature.input_type
        for module in self.modules[1:]:
            if module.type_signature.input_type != first_input_type:
                raise ValueError(
                    f"All parallel modules must have same input type, "
                    f"got {module.type_signature.input_type} != {first_input_type}"
                )
        
        def parallel_transform(x):
            return tuple(module(x) for module in self.modules)
        
        super().__init__(
            name=name,
            arity=1,
            type_signature=TypeSignature(
                input_type=first_input_type,
                output_type="Tuple",
                dtype=self.modules[0].type_signature.dtype
            ),
            shape_constraint=self.modules[0].shape_constraint,
            behavior_traits={
                "differentiable": all(m.behavior_traits.get("differentiable", False) for m in self.modules),
                "non_linear": any(m.behavior_traits.get("non_linear", False) for m in self.modules),
                "stochastic": any(m.behavior_traits.get("stochastic", False) for m in self.modules),
                "stateful": any(m.behavior_traits.get("stateful", False) for m in self.modules),
            },
            transform=parallel_transform,
            topology_type="parallel"
        )
        
        self.operad_contract["meta"] = {
            "container": "parallel",
            "num_branches": len(self.modules)
        }
    
    def __repr__(self) -> str:
        modules_str = " || ".join(m.name for m in self.modules)
        return f"Parallel({modules_str})"


@register_module
class Add(NeuralModule):
    """Add gadget - element-wise addition.
    
    Adds two tensors element-wise with broadcasting support.
    
    Properties:
    - 2-arity operad
    - Linear
    - Type-driven with shared input type constraint
    """
    
    def __init__(self, name: str = "add"):
        """Initialize Add gadget.
        
        Args:
            name: Name of the module
        """
        def add_fn(x, y):
            return x + y
        
        super().__init__(
            name=name,
            arity=2,
            type_signature=TypeSignature(
                input_type="Tensor",
                output_type="Tensor",
                dtype="float32"
            ),
            shape_constraint=ShapeConstraint(
                min_rank=1,
                constraints=["shapes must be broadcastable"]
            ),
            behavior_traits={
                "differentiable": True,
                "non_linear": False,
                "stochastic": False,
                "stateful": False,
            },
            transform=add_fn
        )
        
        self.operad_contract = {
            "arity": 2,
            "input_types": ["Tensor", "Tensor"],
            "output_type": "Tensor",
            "constraints": {"type": "float32", "operation": "element_wise"}
        }


@register_module
class Multiply(NeuralModule):
    """Multiply gadget - element-wise multiplication.
    
    Multiplies two tensors element-wise with broadcasting support.
    
    Properties:
    - 2-arity operad
    - Non-linear
    - Type-driven
    """
    
    def __init__(self, name: str = "multiply"):
        """Initialize Multiply gadget.
        
        Args:
            name: Name of the module
        """
        def mul_fn(x, y):
            return x * y
        
        super().__init__(
            name=name,
            arity=2,
            type_signature=TypeSignature(
                input_type="Tensor",
                output_type="Tensor",
                dtype="float32"
            ),
            shape_constraint=ShapeConstraint(
                min_rank=1,
                constraints=["shapes must be broadcastable"]
            ),
            behavior_traits={
                "differentiable": True,
                "non_linear": True,  # Multiplication is bilinear
                "stochastic": False,
                "stateful": False,
            },
            transform=mul_fn
        )
        
        self.operad_contract = {
            "arity": 2,
            "input_types": ["Tensor", "Tensor"],
            "output_type": "Tensor",
            "constraints": {"type": "float32", "operation": "element_wise"}
        }


@register_module
class Concat(NeuralModule):
    """Concat gadget - concatenation along dimension.
    
    Concatenates n tensors along specified axis with dimension alignment.
    
    Properties:
    - n-arity operad (variable number of inputs)
    - Linear
    - Dimension alignment constraint
    """
    
    def __init__(self, axis: int = -1, name: str = "concat"):
        """Initialize Concat gadget.
        
        Args:
            axis: Axis along which to concatenate
            name: Name of the module
        """
        self.axis = axis
        
        def concat_fn(*tensors):
            return np.concatenate(tensors, axis=self.axis)
        
        # Start with arity 2, but can be adjusted
        super().__init__(
            name=name,
            arity=2,
            type_signature=TypeSignature(
                input_type="Tensor",
                output_type="Tensor",
                dtype="float32"
            ),
            shape_constraint=ShapeConstraint(
                min_rank=1,
                constraints=["all dimensions except concat axis must match"]
            ),
            behavior_traits={
                "differentiable": True,
                "non_linear": False,
                "stochastic": False,
                "stateful": False,
            },
            transform=concat_fn
        )
        
        self.operad_contract = {
            "arity": 2,  # Can be extended
            "input_types": ["Tensor", "Tensor"],
            "output_type": "Tensor",
            "constraints": {
                "type": "float32",
                "shared_input_type": True,
                "dimension_alignment": f"axis={axis}"
            }
        }
    
    def __repr__(self) -> str:
        return f"Concat({self.name}, axis={self.axis})"


@register_module
class Dot(NeuralModule):
    """Dot product gadget - matrix multiplication.
    
    Computes dot product / matrix multiplication between tensors.
    
    Properties:
    - 2-arity operad
    - Non-linear (bilinear)
    - Shape constraint: inner dimensions must match
    """
    
    def __init__(self, name: str = "dot"):
        """Initialize Dot gadget.
        
        Args:
            name: Name of the module
        """
        def dot_fn(x, y):
            # Use appropriate numpy operation based on dimensions
            if x.ndim == 1 and y.ndim == 1:
                return np.dot(x, y)
            else:
                return np.matmul(x, y)
        
        super().__init__(
            name=name,
            arity=2,
            type_signature=TypeSignature(
                input_type="Tensor",
                output_type="Tensor",
                dtype="float32"
            ),
            shape_constraint=ShapeConstraint(
                min_rank=1,
                constraints=["inner dimensions must match for matrix multiplication"]
            ),
            behavior_traits={
                "differentiable": True,
                "non_linear": True,  # Bilinear
                "stochastic": False,
                "stateful": False,
            },
            transform=dot_fn
        )
        
        self.operad_contract = {
            "arity": 2,
            "input_types": ["Tensor", "Tensor"],
            "output_type": "Tensor",
            "constraints": {
                "type": "float32",
                "operation": "matrix_multiplication"
            }
        }
