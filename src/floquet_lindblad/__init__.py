"""Chebyshev-accelerated Arnoldi solver for driven, dissipative systems.

- ``floquet_lindblad.solvers`` — model-agnostic numerics (the :class:`ChebAr` solver,
  its Chebyshev-free counterpart :class:`ArnoldiLindblad`, and the
  thick-restarted :class:`KrylovSchurLindblad`).
- ``floquet_lindblad.models`` — physical model builders (the ATS Hamiltonian in its
  lab-, rotating- and interaction-frame variants).
- ``floquet_lindblad.pipeline`` — one sweep point end to end, shared by the sweep
  scripts and the test notebooks (and imported by cluster workers).
- ``floquet_lindblad.io`` — JSON (de)serialization helpers for sweep results.
"""

from floquet_lindblad.pipeline import escalating_setup, solve_point, solve_point_safe
from floquet_lindblad.solvers.arnoldi_no_cheb import ArnoldiLindblad
from floquet_lindblad.solvers.cheb_ar import ChebAr
from floquet_lindblad.solvers.krylov_schur import KrylovSchurLindblad

__all__ = [
    "ArnoldiLindblad",
    "ChebAr",
    "KrylovSchurLindblad",
    "escalating_setup",
    "solve_point",
    "solve_point_safe",
]
