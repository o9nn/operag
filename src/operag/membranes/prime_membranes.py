"""Prime-Factor Shaped Membrane Topologies.

This module implements P-system inspired computational membranes whose structure
is determined by prime factorization, creating a biomimetic compartmentalization
of computation.
"""

from typing import List, Dict, Any, Set, Optional, Tuple
import numpy as np
from functools import reduce
from operator import mul


def prime_factorization(n: int) -> List[int]:
    """Compute prime factorization of a number.
    
    Args:
        n: Integer to factorize
        
    Returns:
        List of prime factors (with repetition)
    """
    if n < 2:
        return []
    
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


class PrimeFactorMembrane:
    """A computational membrane whose topology is shaped by prime factorization.
    
    The membrane structure creates nested compartments based on the prime factors
    of its size parameter, implementing a hierarchical P-system where:
    - Each prime factor defines a compartment level
    - Data flows through topologically-constrained paths
    - Permissions are encoded in membrane boundaries
    """
    
    def __init__(self, size: int, label: Optional[str] = None):
        """Initialize a prime-factor shaped membrane.
        
        Args:
            size: The size parameter that determines membrane topology via prime factorization
            label: Optional label for this membrane
        """
        self.size = size
        self.label = label or f"M_{size}"
        self.prime_factors = prime_factorization(size)
        self.structure = self._build_structure()
        self.contents: Dict[str, Any] = {}
        self.permissions: Set[str] = set()
        
    def _build_structure(self) -> Dict[str, Any]:
        """Build the membrane structure based on prime factorization.
        
        Returns:
            Dictionary representing the hierarchical membrane structure
        """
        if not self.prime_factors:
            return {"level": 0, "factor": None, "children": []}
        
        # Build nested structure from prime factors
        structure = {"level": 0, "factor": self.prime_factors[0], "children": []}
        current = structure
        
        for i, factor in enumerate(self.prime_factors[1:], 1):
            child = {"level": i, "factor": factor, "children": []}
            current["children"].append(child)
            current = child
            
        return structure
    
    def get_topology_signature(self) -> Tuple[int, ...]:
        """Get the topological signature of this membrane.
        
        Returns:
            Tuple of prime factors representing the topology
        """
        return tuple(self.prime_factors)
    
    def store(self, key: str, value: Any, permission: Optional[str] = None):
        """Store data in the membrane with optional permission gating.
        
        Args:
            key: Key to store data under
            value: Value to store
            permission: Optional permission required to access this data
        """
        self.contents[key] = value
        if permission:
            self.permissions.add(permission)
    
    def retrieve(self, key: str, permission: Optional[str] = None) -> Optional[Any]:
        """Retrieve data from the membrane with permission checking.
        
        Args:
            key: Key to retrieve
            permission: Permission token for access
            
        Returns:
            Retrieved value or None if not found/unauthorized
        """
        if key not in self.contents:
            return None
        
        # Check if permission is required and provided
        if self.permissions and permission not in self.permissions:
            return None
        
        return self.contents[key]
    
    def can_merge_with(self, other: 'PrimeFactorMembrane') -> bool:
        """Check if this membrane can topologically merge with another.
        
        Two membranes can merge if their prime factorizations are compatible
        (share common factors or are coprime).
        
        Args:
            other: Another membrane to check compatibility with
            
        Returns:
            True if membranes can merge
        """
        # Check for common prime factors or coprimality
        self_factors = set(self.prime_factors)
        other_factors = set(other.prime_factors)
        
        # Can merge if they share factors or are coprime
        common = self_factors & other_factors
        return len(common) > 0 or (len(common) == 0 and 
                                   np.gcd(self.size, other.size) == 1)
    
    def merge(self, other: 'PrimeFactorMembrane') -> 'PrimeFactorMembrane':
        """Merge this membrane with another, creating a new composite membrane.
        
        Args:
            other: Membrane to merge with
            
        Returns:
            New merged membrane
            
        Raises:
            ValueError: If membranes cannot be merged topologically
        """
        if not self.can_merge_with(other):
            raise ValueError(f"Membranes {self.label} and {other.label} cannot merge topologically")
        
        # Create new membrane with combined size
        new_size = self.size * other.size
        merged = PrimeFactorMembrane(new_size, f"{self.label}⊗{other.label}")
        
        # Merge contents
        merged.contents.update(self.contents)
        merged.contents.update(other.contents)
        
        # Merge permissions
        merged.permissions = self.permissions | other.permissions
        
        return merged
    
    def get_depth(self) -> int:
        """Get the depth of the membrane hierarchy.
        
        Returns:
            Number of nested levels (equals number of prime factors)
        """
        return len(self.prime_factors)
    
    def __repr__(self) -> str:
        return f"PrimeFactorMembrane(size={self.size}, factors={self.prime_factors}, label={self.label})"
    
    def __str__(self) -> str:
        factors_str = " × ".join(map(str, self.prime_factors))
        return f"{self.label}: {self.size} = {factors_str if factors_str else '1'} (depth={self.get_depth()})"
