"""Tests for Operad Gadget Constellations."""

import pytest
import numpy as np
from operag.operad import OperadGadget, GadgetConstellation


class TestOperadGadget:
    """Tests for OperadGadget class."""
    
    def test_initialization(self):
        g = OperadGadget("add", arity=2)
        assert g.name == "add"
        assert g.arity == 2
        assert g.output_dim == 1
    
    def test_default_transform(self):
        g = OperadGadget("sum", arity=3)
        result = g.apply(1, 2, 3)
        assert result == 6  # Default is sum
    
    def test_custom_transform(self):
        g = OperadGadget("multiply", arity=2, transform=lambda x, y: x * y)
        assert g.apply(3, 4) == 12
    
    def test_callable(self):
        g = OperadGadget("add", arity=2, transform=lambda x, y: x + y)
        assert g(5, 7) == 12
    
    def test_can_compose_with(self):
        g1 = OperadGadget("f", arity=3, topology_type="sequential")
        g2 = OperadGadget("g", arity=2, topology_type="sequential")
        
        assert g1.can_compose_with(g2, position=0)
        assert g1.can_compose_with(g2, position=1)
        assert not g1.can_compose_with(g2, position=5)  # Invalid position
    
    def test_compose_simple(self):
        add = OperadGadget("add", arity=2, transform=lambda x, y: x + y)
        mul = OperadGadget("mul", arity=2, transform=lambda x, y: x * y)
        
        # Compose: add(mul(x,y), z) where z is passed through
        composed = add.compose([mul], positions=[0])
        
        # mul(3, 4) + 5 = 12 + 5 = 17
        result = composed(3, 4, 5)
        assert result == 17
    
    def test_compose_multiple(self):
        add = OperadGadget("add", arity=2, transform=lambda x, y: x + y)
        mul = OperadGadget("mul", arity=2, transform=lambda x, y: x * y)
        scale = OperadGadget("scale", arity=1, transform=lambda x: x * 2)
        
        # add(mul(x,y), scale(z))
        composed = add.compose([mul, scale], positions=[0, 1])
        
        # mul(3, 4) + scale(5) = 12 + 10 = 22
        result = composed(3, 4, 5)
        assert result == 22
    
    def test_arity_mismatch(self):
        g = OperadGadget("f", arity=2)
        
        with pytest.raises(ValueError):
            g.apply(1, 2, 3)  # Too many inputs


class TestGadgetConstellation:
    """Tests for GadgetConstellation class."""
    
    def test_initialization(self):
        c = GadgetConstellation("test")
        assert c.name == "test"
        assert len(c.gadgets) == 0
    
    def test_add_gadget(self):
        c = GadgetConstellation()
        g = OperadGadget("add", arity=2)
        c.add_gadget(g)
        
        assert "add" in c.gadgets
        assert c.gadgets["add"] is g
    
    def test_connect(self):
        c = GadgetConstellation()
        g1 = OperadGadget("f", arity=2, topology_type="sequential")
        g2 = OperadGadget("g", arity=2, topology_type="sequential")
        
        c.add_gadget(g1)
        c.add_gadget(g2)
        c.connect("g", "f", position=0)
        
        assert ("g", 0) in c.composition_graph["f"]
    
    def test_connect_invalid_source(self):
        c = GadgetConstellation()
        g = OperadGadget("f", arity=2)
        c.add_gadget(g)
        
        with pytest.raises(ValueError):
            c.connect("nonexistent", "f")
    
    def test_connect_invalid_target(self):
        c = GadgetConstellation()
        g = OperadGadget("f", arity=2)
        c.add_gadget(g)
        
        with pytest.raises(ValueError):
            c.connect("f", "nonexistent")
    
    def test_topology_signature(self):
        c = GadgetConstellation()
        g1 = OperadGadget("f", arity=2, topology_type="sequential")
        g2 = OperadGadget("g", arity=3, topology_type="tree")
        
        c.add_gadget(g1)
        c.add_gadget(g2)
        c.connect("f", "g", position=0)
        
        sig = c.get_topology_signature()
        assert sig["num_gadgets"] == 2
        assert sig["num_connections"] == 1
        assert sig["total_arity"] == 5
        assert sig["max_arity"] == 3
    
    def test_compute_depth(self):
        c = GadgetConstellation()
        g1 = OperadGadget("a", arity=1, topology_type="sequential")
        g2 = OperadGadget("b", arity=1, topology_type="sequential")
        g3 = OperadGadget("c", arity=1, topology_type="sequential")
        
        c.add_gadget(g1)
        c.add_gadget(g2)
        c.add_gadget(g3)
        
        # Linear chain: a -> b -> c
        c.connect("a", "b", position=0)
        c.connect("b", "c", position=0)
        
        assert c.compute_depth() == 3
    
    def test_visualize_topology(self):
        c = GadgetConstellation("example")
        g = OperadGadget("add", arity=2)
        c.add_gadget(g)
        
        viz = c.visualize_topology()
        assert "Constellation: example" in viz
        assert "add" in viz
    
    def test_empty_constellation_depth(self):
        c = GadgetConstellation()
        assert c.compute_depth() == 0
