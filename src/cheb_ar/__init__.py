"""Chebyshev-accelerated Arnoldi solver for driven, dissipative systems.

- ``cheb_ar.solvers`` — model-agnostic numerics (the :class:`ChebAr` solver).
- ``cheb_ar.models`` — physical model builders (to be populated; the ATS
  Hamiltonian builders currently still live in scripts and notebooks).
"""

from cheb_ar.solvers.cheb_ar import ChebAr

__all__ = ["ChebAr"]
