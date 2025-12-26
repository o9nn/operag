"""Neural Network Operad Gadgets.

This module implements neural network components as typed operadic gadgets,
integrating typed operadic composition, graph-based functional abstraction,
and agent-modular logic.
"""

from .base import NeuralModule, TypeSignature, ShapeConstraint, register_module, get_registered_modules
from .containers import Sequential, Parallel, Add, Multiply, Concat, Dot
from .activations import ReLU, Tanh, Sigmoid, Softmax, LeakyReLU
from .loss import MSELoss, CrossEntropyLoss, BCELoss, L1Loss
from .layers import Reshape, Flatten, Identity, Mean, Max, Sum

__all__ = [
    # Base
    "NeuralModule",
    "TypeSignature",
    "ShapeConstraint",
    "register_module",
    "get_registered_modules",
    # Containers
    "Sequential",
    "Parallel",
    "Add",
    "Multiply",
    "Concat",
    "Dot",
    # Activations
    "ReLU",
    "Tanh",
    "Sigmoid",
    "Softmax",
    "LeakyReLU",
    # Loss functions
    "MSELoss",
    "CrossEntropyLoss",
    "BCELoss",
    "L1Loss",
    # Layers
    "Reshape",
    "Flatten",
    "Identity",
    "Mean",
    "Max",
    "Sum",
]
