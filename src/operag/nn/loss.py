"""Loss functions as terminal operad gadgets.

This module implements loss functions (criteria) as terminal operads with
special "result" morphism and error signal tracking.
"""

import numpy as np
from typing import Dict, Any, Optional
from .base import NeuralModule, TypeSignature, ShapeConstraint, register_module


@register_module
class MSELoss(NeuralModule):
    """Mean Squared Error loss function.
    
    Computes: MSE = mean((prediction - target)^2)
    
    Properties:
    - 2-arity operad (prediction, target)
    - Terminal operad (outputs scalar loss)
    - Differentiable
    - Tracks gradient origin
    """
    
    def __init__(self, name: str = "mse_loss"):
        """Initialize MSE loss.
        
        Args:
            name: Name of the module
        """
        def mse_fn(prediction, target):
            return np.mean((prediction - target) ** 2)
        
        super().__init__(
            name=name,
            arity=2,  # Takes prediction and target
            type_signature=TypeSignature(
                input_type="Tensor",
                output_type="Scalar",
                dtype="float32"
            ),
            shape_constraint=ShapeConstraint(
                min_rank=1
            ),
            behavior_traits={
                "differentiable": True,
                "non_linear": False,  # MSE is quadratic but treated as linear
                "stochastic": False,
                "stateful": False,
            },
            transform=mse_fn
        )
        
        # Terminal operad contract
        self.operad_contract = {
            "arity": 2,
            "input_types": ["Tensor", "Tensor"],
            "output_type": "Scalar",
            "terminal": True,
            "loss_type": "regression"
        }
        
        # Error signal tracking
        self.error_signals = []
        self.gradient_trace = []
    
    def forward(self, prediction, target):
        """Forward pass with error signal tracking.
        
        Args:
            prediction: Predicted values
            target: Ground truth values
            
        Returns:
            Scalar loss value
        """
        loss = super().forward(prediction, target)
        
        # Track error signal
        self.error_signals.append({
            "loss_value": float(loss),
            "prediction_shape": prediction.shape if isinstance(prediction, np.ndarray) else None,
            "target_shape": target.shape if isinstance(target, np.ndarray) else None,
        })
        
        return loss
    
    def get_error_signals(self) -> list:
        """Get tracked error signals."""
        return self.error_signals
    
    def clear_error_signals(self):
        """Clear error signal history."""
        self.error_signals = []
        self.gradient_trace = []


@register_module
class CrossEntropyLoss(NeuralModule):
    """Cross-Entropy loss function for classification.
    
    Computes: CE = -sum(target * log(prediction))
    
    Properties:
    - 2-arity operad (prediction, target)
    - Terminal operad (outputs scalar loss)
    - Differentiable
    - For classification tasks
    """
    
    def __init__(self, name: str = "cross_entropy_loss"):
        """Initialize Cross-Entropy loss.
        
        Args:
            name: Name of the module
        """
        def ce_fn(prediction, target):
            # Add small epsilon for numerical stability
            epsilon = 1e-7
            prediction = np.clip(prediction, epsilon, 1 - epsilon)
            return -np.mean(np.sum(target * np.log(prediction), axis=-1))
        
        super().__init__(
            name=name,
            arity=2,
            type_signature=TypeSignature(
                input_type="Tensor",
                output_type="Scalar",
                dtype="float32"
            ),
            shape_constraint=ShapeConstraint(
                min_rank=2  # Typically [batch_size, num_classes]
            ),
            behavior_traits={
                "differentiable": True,
                "non_linear": True,
                "stochastic": False,
                "stateful": False,
            },
            transform=ce_fn
        )
        
        self.operad_contract = {
            "arity": 2,
            "input_types": ["Tensor", "Tensor"],
            "output_type": "Scalar",
            "terminal": True,
            "loss_type": "classification"
        }
        
        self.error_signals = []
        self.gradient_trace = []
    
    def forward(self, prediction, target):
        """Forward pass with error signal tracking."""
        loss = super().forward(prediction, target)
        
        self.error_signals.append({
            "loss_value": float(loss),
            "prediction_shape": prediction.shape if isinstance(prediction, np.ndarray) else None,
            "target_shape": target.shape if isinstance(target, np.ndarray) else None,
        })
        
        return loss
    
    def get_error_signals(self) -> list:
        """Get tracked error signals."""
        return self.error_signals
    
    def clear_error_signals(self):
        """Clear error signal history."""
        self.error_signals = []
        self.gradient_trace = []


@register_module
class BCELoss(NeuralModule):
    """Binary Cross-Entropy loss function.
    
    Computes: BCE = -mean(target * log(prediction) + (1-target) * log(1-prediction))
    
    Properties:
    - 2-arity operad (prediction, target)
    - Terminal operad (outputs scalar loss)
    - Differentiable
    - For binary classification
    """
    
    def __init__(self, name: str = "bce_loss"):
        """Initialize BCE loss.
        
        Args:
            name: Name of the module
        """
        def bce_fn(prediction, target):
            # Add small epsilon for numerical stability
            epsilon = 1e-7
            prediction = np.clip(prediction, epsilon, 1 - epsilon)
            return -np.mean(
                target * np.log(prediction) + (1 - target) * np.log(1 - prediction)
            )
        
        super().__init__(
            name=name,
            arity=2,
            type_signature=TypeSignature(
                input_type="Tensor",
                output_type="Scalar",
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
            transform=bce_fn
        )
        
        self.operad_contract = {
            "arity": 2,
            "input_types": ["Tensor", "Tensor"],
            "output_type": "Scalar",
            "terminal": True,
            "loss_type": "binary_classification"
        }
        
        self.error_signals = []
        self.gradient_trace = []
    
    def forward(self, prediction, target):
        """Forward pass with error signal tracking."""
        loss = super().forward(prediction, target)
        
        self.error_signals.append({
            "loss_value": float(loss),
            "prediction_shape": prediction.shape if isinstance(prediction, np.ndarray) else None,
            "target_shape": target.shape if isinstance(target, np.ndarray) else None,
        })
        
        return loss
    
    def get_error_signals(self) -> list:
        """Get tracked error signals."""
        return self.error_signals
    
    def clear_error_signals(self):
        """Clear error signal history."""
        self.error_signals = []
        self.gradient_trace = []


@register_module
class L1Loss(NeuralModule):
    """L1 (Mean Absolute Error) loss function.
    
    Computes: L1 = mean(|prediction - target|)
    
    Properties:
    - 2-arity operad (prediction, target)
    - Terminal operad (outputs scalar loss)
    - Not differentiable at zero (but used in practice)
    - Robust to outliers
    """
    
    def __init__(self, name: str = "l1_loss"):
        """Initialize L1 loss.
        
        Args:
            name: Name of the module
        """
        def l1_fn(prediction, target):
            return np.mean(np.abs(prediction - target))
        
        super().__init__(
            name=name,
            arity=2,
            type_signature=TypeSignature(
                input_type="Tensor",
                output_type="Scalar",
                dtype="float32"
            ),
            shape_constraint=ShapeConstraint(
                min_rank=1
            ),
            behavior_traits={
                "differentiable": False,  # Not differentiable at zero
                "non_linear": False,
                "stochastic": False,
                "stateful": False,
            },
            transform=l1_fn
        )
        
        self.operad_contract = {
            "arity": 2,
            "input_types": ["Tensor", "Tensor"],
            "output_type": "Scalar",
            "terminal": True,
            "loss_type": "regression"
        }
        
        self.error_signals = []
    
    def forward(self, prediction, target):
        """Forward pass with error signal tracking."""
        loss = super().forward(prediction, target)
        
        self.error_signals.append({
            "loss_value": float(loss),
            "prediction_shape": prediction.shape if isinstance(prediction, np.ndarray) else None,
            "target_shape": target.shape if isinstance(target, np.ndarray) else None,
        })
        
        return loss
    
    def get_error_signals(self) -> list:
        """Get tracked error signals."""
        return self.error_signals
    
    def clear_error_signals(self):
        """Clear error signal history."""
        self.error_signals = []
