"""Bit-flip rates of driven, dissipative systems from the one-period propagator.

The rate is the leading eigenvalue of the trace-projected one-period
propagator of the Lindblad master equation. The main solver is the
thick-restarted Arnoldi :class:`KrylovSchurLindblad`; the Chebyshev-filtered
:class:`ChebAr` and the plain :class:`ArnoldiLindblad` are kept as secondary
methods and cross-checks.

- ``floquet_lindblad.solvers`` — model-agnostic numerics (the three solvers).
- ``floquet_lindblad.models`` — physical model builders (the ATS Hamiltonian in its
  lab-, rotating- and interaction-frame variants).
- ``floquet_lindblad.pipeline`` — one sweep point end to end, one module per
  solver (:func:`solve_point_ks` main, :func:`solve_point_cheb_ar`), shared by
  the sweep scripts and notebooks and imported by cluster workers.
- ``floquet_lindblad.io`` — JSON (de)serialization helpers for sweep results.
"""

from floquet_lindblad.pipeline import (
    escalating_setup,
    solve_point_cheb_ar,
    solve_point_cheb_ar_safe,
    solve_point_ks,
    solve_point_ks_safe,
)
from floquet_lindblad.solvers.arnoldi_no_cheb import ArnoldiLindblad
from floquet_lindblad.solvers.cheb_ar import ChebAr
from floquet_lindblad.solvers.krylov_schur import KrylovSchurLindblad

__all__ = [
    "KrylovSchurLindblad",
    "solve_point_ks",
    "solve_point_ks_safe",
    "ArnoldiLindblad",
    "ChebAr",
    "escalating_setup",
    "solve_point_cheb_ar",
    "solve_point_cheb_ar_safe",
]
