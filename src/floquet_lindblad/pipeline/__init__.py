"""One sweep point end to end, one module per solver.

- :mod:`~floquet_lindblad.pipeline.krylov_schur` — the main pipeline
  (:func:`solve_point_ks`, thick-restarted Arnoldi on the one-period
  propagator).
- :mod:`~floquet_lindblad.pipeline.cheb_ar` — the Chebyshev-filtered
  ``ChebAr`` pipeline (:func:`solve_point_cheb_ar`), kept as a secondary
  method and a cross-check.

Both share the calling contract (sweep coordinates, ``x0`` / ``V_prev`` warm
start, ``want_x_ritz`` / ``want_V``) and return plain numpy / python, so
results can cross to a client without the GPU stack. The names carry the
solver on purpose: there is no bare ``solve_point``, so code written against
one solver cannot silently run the other.

They live in the package rather than in the scripts because cluster workers
import them: the drivers ship ``floquet_lindblad`` through the Ray
``runtime_env`` and each task calls one of the ``*_safe`` functions.
"""

from floquet_lindblad.pipeline.cheb_ar import (
    escalating_setup,
    solve_point_cheb_ar,
    solve_point_cheb_ar_safe,
)
from floquet_lindblad.pipeline.common import error_entry
from floquet_lindblad.pipeline.krylov_schur import solve_point_ks, solve_point_ks_safe

__all__ = [
    "error_entry",
    "escalating_setup",
    "solve_point_cheb_ar",
    "solve_point_cheb_ar_safe",
    "solve_point_ks",
    "solve_point_ks_safe",
]
