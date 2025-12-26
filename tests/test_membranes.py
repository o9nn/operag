"""Tests for Prime-Factor Shaped Membrane Topologies."""

import pytest
import numpy as np
from operag.membranes import PrimeFactorMembrane, prime_factorization


class TestPrimeFactorization:
    """Tests for prime factorization helper function."""
    
    def test_small_primes(self):
        assert prime_factorization(2) == [2]
        assert prime_factorization(3) == [3]
        assert prime_factorization(5) == [5]
    
    def test_composite_numbers(self):
        assert prime_factorization(4) == [2, 2]
        assert prime_factorization(6) == [2, 3]
        assert prime_factorization(12) == [2, 2, 3]
        assert prime_factorization(30) == [2, 3, 5]
    
    def test_edge_cases(self):
        assert prime_factorization(1) == []
        assert prime_factorization(0) == []


class TestPrimeFactorMembrane:
    """Tests for PrimeFactorMembrane class."""
    
    def test_initialization(self):
        m = PrimeFactorMembrane(12, "test")
        assert m.size == 12
        assert m.label == "test"
        assert m.prime_factors == [2, 2, 3]
        assert m.get_depth() == 3
    
    def test_topology_signature(self):
        m = PrimeFactorMembrane(30)
        assert m.get_topology_signature() == (2, 3, 5)
    
    def test_store_and_retrieve(self):
        m = PrimeFactorMembrane(6)
        m.store("key1", "value1")
        assert m.retrieve("key1") == "value1"
        assert m.retrieve("nonexistent") is None
    
    def test_permission_gating(self):
        m = PrimeFactorMembrane(6)
        m.store("secret", "data", permission="admin")
        
        # Without permission
        assert m.retrieve("secret") is None
        
        # With permission
        assert m.retrieve("secret", permission="admin") == "data"
    
    def test_can_merge_with(self):
        m1 = PrimeFactorMembrane(6)  # 2 × 3
        m2 = PrimeFactorMembrane(10)  # 2 × 5
        m3 = PrimeFactorMembrane(15)  # 3 × 5
        
        # Share factor 2
        assert m1.can_merge_with(m2)
        
        # Share factor 3
        assert m1.can_merge_with(m3)
        
        # Share factor 5
        assert m2.can_merge_with(m3)
    
    def test_merge(self):
        m1 = PrimeFactorMembrane(6, "M6")
        m2 = PrimeFactorMembrane(10, "M10")
        
        m1.store("data1", [1, 2, 3])
        m2.store("data2", [4, 5, 6])
        
        merged = m1.merge(m2)
        
        assert merged.size == 60
        assert merged.retrieve("data1") == [1, 2, 3]
        assert merged.retrieve("data2") == [4, 5, 6]
        assert "M6" in merged.label and "M10" in merged.label
    
    def test_merge_incompatible(self):
        m1 = PrimeFactorMembrane(7)  # Prime
        m2 = PrimeFactorMembrane(11)  # Different prime
        
        # Two different primes are coprime, so they can merge
        assert m1.can_merge_with(m2)
    
    def test_structure_building(self):
        m = PrimeFactorMembrane(12)
        assert m.structure["level"] == 0
        assert m.structure["factor"] == 2
        assert len(m.structure["children"]) > 0
