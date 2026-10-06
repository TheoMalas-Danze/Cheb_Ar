"""What a model hands to a pipeline — the contract that keeps pipelines model-agnostic.

A pipeline asks a builder for a :class:`Lindbladian` and never looks inside the
model: the builder decides the frame, and the optional fields say what the
solvers can exploit (``jump_ops_LdL``, ``output_phase``) and how to get back to
the Fock basis (``V``).
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# `transform_vectorized_state` returns a jax array; keep it complex128 even
# when nothing else has switched x64 on yet.
jax.config.update("jax_enable_x64", True)


class Lindbladian(NamedTuple):
    """A Floquet Lindbladian, ready for ``KrylovSchurLindblad`` / ``ChebAr``.

    Attributes
    ----------
    H : dynamiqs operator
        Hamiltonian (time-dependent, periodic in ``T_block``).
    jump_ops : list
        Lindblad jump operators.
    T_block : float
        Duration of one Floquet block.
    params : dict
        Derived quantities, stored alongside every sweep result.
    jump_ops_LdL : list, optional
        Precomputed ``L^dag L`` operators, for ``dq.mesolve_fast``.
    output_phase : array (N,), optional
        Per-basis-state phase undoing the frame rotation after one block.
    V : array (N, N), optional
        Frame transformation: the state is written in the basis ``V``, i.e.
        ``rho_fock = V @ rho @ V^dag``. ``None`` when the state is already in
        the Fock basis (lab and rotating frames).
    """

    H: Any
    jump_ops: list
    T_block: float
    params: dict
    jump_ops_LdL: list | None = None
    output_phase: Any = None
    V: Any = None


def transform_vectorized_state(x_vec, V_from, V_to):
    """Re-express a vectorized operator between two interaction-frame bases.

    ``x_vec`` is a column-major vectorized operator written in the eigenbasis
    ``V_from`` (as returned by an interaction-frame builder); the result is the
    same operator written in the eigenbasis ``V_to``. Used to warm-start a
    sweep point from the previous point's Ritz vector when the static
    Hamiltonian — hence its eigenbasis — changes along the sweep (e.g. an
    ``epsilon_p`` sweep; for an ``alpha_sq`` sweep the basis is unchanged and
    this reduces to the identity).
    """
    N = V_from.shape[0]
    rho = np.asarray(x_vec).reshape((N, N), order="F")
    rho_lab = V_from @ rho @ V_from.conj().T
    rho_to = V_to.conj().T @ rho_lab @ V_to
    return jnp.array(rho_to.reshape(-1, order="F"))
