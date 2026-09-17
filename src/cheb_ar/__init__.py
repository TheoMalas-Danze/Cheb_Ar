"""Chebyshev-accelerated Arnoldi solver for driven, dissipative systems.

- ``cheb_ar.solvers`` — model-agnostic numerics (the :class:`ChebAr` solver).
- ``cheb_ar.models`` — physical model builders (the ATS Hamiltonian in its
  lab-, rotating- and interaction-frame variants).
- ``cheb_ar.pipeline`` — one sweep point end to end, shared by the sweep
  scripts and the test notebooks (and imported by cluster workers).
- ``cheb_ar.io`` — JSON (de)serialization helpers for sweep results.
"""

from cheb_ar.pipeline import escalating_setup, solve_point, solve_point_safe
from cheb_ar.solvers.cheb_ar import ChebAr

__all__ = ["ChebAr", "escalating_setup", "solve_point", "solve_point_safe"]
