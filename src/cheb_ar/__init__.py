"""Chebyshev-accelerated Arnoldi solver for driven, dissipative systems.

- ``cheb_ar.solvers`` — model-agnostic numerics (the :class:`ChebAr` solver).
- ``cheb_ar.models`` — physical model builders (the ATS Hamiltonian in its
  lab-, rotating- and interaction-frame variants).
- ``cheb_ar.io`` — JSON (de)serialization helpers for sweep results.
"""

from cheb_ar.solvers.cheb_ar import ChebAr

__all__ = ["ChebAr"]
