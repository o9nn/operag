"""Operad Gadget Constellations.

This module implements operadic structures for composable computational "gadgets".
Operads provide the mathematical framework for composing operations with multiple
inputs and outputs in a topologically meaningful way.
"""

from typing import List, Dict, Callable, Any, Optional, Tuple
import numpy as np
from functools import reduce


class OperadGadget:
    """A computational gadget that can be composed operadically.
    
    A gadget is a topological unit of computation with:
    - Input topology (arity)
    - Output topology 
    - A transformation function
    - Topological constraints on composition
    """
    
    def __init__(self, 
                 name: str,
                 arity: int,
                 output_dim: int = 1,
                 transform: Optional[Callable] = None,
                 topology_type: str = "sequential"):
        """Initialize an operad gadget.
        
        Args:
            name: Name of the gadget
            arity: Number of inputs (input topology)
            output_dim: Dimension of output
            transform: Transformation function to apply
            topology_type: Type of topological structure ("sequential", "parallel", "tree")
        """
        self.name = name
        self.arity = arity
        self.output_dim = output_dim
        self.topology_type = topology_type
        
        # Default transform is sum
        self.transform = transform or (lambda *args: np.sum(args, axis=0))
        
        # Composition compatibility
        self.input_signature: Optional[Tuple] = None
        self.output_signature: Optional[Tuple] = None
        
    def can_compose_with(self, other: 'OperadGadget', position: int = 0) -> bool:
        """Check if this gadget can be composed with another.
        
        Args:
            other: Another gadget to compose with
            position: Position where to insert the other gadget
            
        Returns:
            True if composition is topologically valid
        """
        # Check if position is valid
        if position < 0 or position >= self.arity:
            return False
        
        # Check topological compatibility
        if self.topology_type == "sequential" and other.topology_type == "sequential":
            return True
        
        if self.topology_type == "tree" and other.topology_type in ["tree", "sequential"]:
            return True
        
        return False
    
    def compose(self, others: List['OperadGadget'], positions: Optional[List[int]] = None) -> 'OperadGadget':
        """Compose this gadget with others using operadic composition.
        
        Args:
            others: List of gadgets to compose into this one
            positions: Positions where to insert each gadget (default: sequential)
            
        Returns:
            New composed gadget
        """
        if positions is None:
            positions = list(range(len(others)))
        
        if len(others) != len(positions):
            raise ValueError("Number of gadgets must match number of positions")
        
        # Check all compositions are valid
        for gadget, pos in zip(others, positions):
            if not self.can_compose_with(gadget, pos):
                raise ValueError(f"Cannot compose {gadget.name} at position {pos}")
        
        # Create new composed gadget
        new_arity = self.arity - len(others) + sum(g.arity for g in others)
        composed_name = f"{self.name}∘({','.join(g.name for g in others)})"
        
        # Create composed transformation
        def composed_transform(*args):
            # Distribute inputs to composed gadgets
            results = []
            arg_idx = 0
            pos_idx = 0
            
            for i in range(self.arity):
                if pos_idx < len(positions) and i == positions[pos_idx]:
                    # Apply the composed gadget
                    gadget = others[pos_idx]
                    gadget_inputs = args[arg_idx:arg_idx + gadget.arity]
                    results.append(gadget.transform(*gadget_inputs))
                    arg_idx += gadget.arity
                    pos_idx += 1
                else:
                    # Pass through input
                    if arg_idx < len(args):
                        results.append(args[arg_idx])
                        arg_idx += 1
            
            # Apply this gadget's transform to results
            return self.transform(*results)
        
        return OperadGadget(
            name=composed_name,
            arity=new_arity,
            output_dim=self.output_dim,
            transform=composed_transform,
            topology_type=self.topology_type
        )
    
    def apply(self, *inputs) -> Any:
        """Apply the gadget transformation to inputs.
        
        Args:
            *inputs: Input values (must match arity)
            
        Returns:
            Transformed output
        """
        if len(inputs) != self.arity:
            raise ValueError(f"Expected {self.arity} inputs, got {len(inputs)}")
        
        return self.transform(*inputs)
    
    def __call__(self, *inputs) -> Any:
        """Make gadget callable."""
        return self.apply(*inputs)
    
    def __repr__(self) -> str:
        return f"OperadGadget({self.name}, arity={self.arity}, type={self.topology_type})"


class GadgetConstellation:
    """A constellation of operad gadgets forming a computational topology.
    
    This represents a complex computation as a network of composed gadgets,
    where the topology determines the flow and transformation of data.
    """
    
    def __init__(self, name: str = "constellation"):
        """Initialize a gadget constellation.
        
        Args:
            name: Name of the constellation
        """
        self.name = name
        self.gadgets: Dict[str, OperadGadget] = {}
        self.composition_graph: Dict[str, List[Tuple[str, int]]] = {}
        
    def add_gadget(self, gadget: OperadGadget):
        """Add a gadget to the constellation.
        
        Args:
            gadget: Gadget to add
        """
        self.gadgets[gadget.name] = gadget
        if gadget.name not in self.composition_graph:
            self.composition_graph[gadget.name] = []
    
    def connect(self, source: str, target: str, position: int = 0):
        """Connect two gadgets in the constellation.
        
        Args:
            source: Name of source gadget
            target: Name of target gadget
            position: Position in target's inputs
        """
        if source not in self.gadgets:
            raise ValueError(f"Source gadget {source} not found")
        if target not in self.gadgets:
            raise ValueError(f"Target gadget {target} not found")
        
        # Check if connection is topologically valid
        if not self.gadgets[target].can_compose_with(self.gadgets[source], position):
            raise ValueError(f"Cannot connect {source} to {target} at position {position}")
        
        if target not in self.composition_graph:
            self.composition_graph[target] = []
        
        self.composition_graph[target].append((source, position))
    
    def get_topology_signature(self) -> Dict[str, Any]:
        """Get the topological signature of the constellation.
        
        Returns:
            Dictionary describing the topology
        """
        return {
            "num_gadgets": len(self.gadgets),
            "num_connections": sum(len(conns) for conns in self.composition_graph.values()),
            "total_arity": sum(g.arity for g in self.gadgets.values()),
            "gadget_types": {g.topology_type for g in self.gadgets.values()},
            "max_arity": max((g.arity for g in self.gadgets.values()), default=0),
        }
    
    def compute_depth(self) -> int:
        """Compute the maximum composition depth of the constellation.
        
        Returns:
            Maximum depth
        """
        def compute_node_depth(gadget_name: str, visited: set) -> int:
            if gadget_name in visited:
                return 0
            
            visited.add(gadget_name)
            
            if gadget_name not in self.composition_graph or not self.composition_graph[gadget_name]:
                return 1
            
            depths = []
            for source, _ in self.composition_graph[gadget_name]:
                depths.append(compute_node_depth(source, visited.copy()))
            
            return 1 + max(depths, default=0)
        
        if not self.gadgets:
            return 0
        
        return max(compute_node_depth(name, set()) for name in self.gadgets)
    
    def visualize_topology(self) -> str:
        """Create a text visualization of the constellation topology.
        
        Returns:
            String representation of the topology
        """
        lines = [f"Constellation: {self.name}"]
        lines.append(f"Gadgets: {len(self.gadgets)}")
        lines.append(f"Depth: {self.compute_depth()}")
        lines.append("\nGadget Network:")
        
        for gadget_name in self.gadgets:
            gadget = self.gadgets[gadget_name]
            line = f"  {gadget_name} (arity={gadget.arity}, type={gadget.topology_type})"
            
            if gadget_name in self.composition_graph and self.composition_graph[gadget_name]:
                inputs = ", ".join(f"{src}@{pos}" for src, pos in self.composition_graph[gadget_name])
                line += f" ← [{inputs}]"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"GadgetConstellation({self.name}, gadgets={len(self.gadgets)})"
