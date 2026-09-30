"""Full-ATS (Asymmetrically Threaded SQUID) model builders.

Three variants of the same driven, dissipative two-mode system (storage mode
``a``, buffer mode ``b``), differing only in the frame:

- :func:`build_ats_hamiltonian` — lab frame (``H_0`` + modulated drive at
  ``w_b`` + ``sinm`` nonlinearity).
- :func:`build_ats_hamiltonian_rotating` — rotating frame (modulated
  ``a``/``b`` operators, static drive, no ``H_0``); previously named
  ``build_ats_hamiltonian_r``.
- :func:`build_ats_hamiltonian_interaction` — interaction frame of the static
  Hamiltonian; additionally returns the precomputed ``L^dag L`` jump operators
  (for ``dq.mesolve_fast``), the per-block ``output_phase`` and the frame
  transformation ``V``, pairing with the extended ``ChebAr`` solver.

The physical constants below are the single source of truth for the default
experimental parameters (Josephson energy, phases, frequencies); sweep scripts
should import them rather than redefine them. Historical note: the rotating
variant was extracted with a default ``epsilon_p=0.3`` while the other two use
``0.1`` — the per-variant defaults are preserved as found.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import jv

jax.config.update("jax_enable_x64", True)

import dynamiqs as dq

dq.set_precision("double")

# --- Default experimental parameters (single source of truth) --- #
W_A = 25.338776456203686  # storage-mode frequency
KAPPA_B = 5 / 10.4  # buffer single-photon loss rate
E_J = 37 * 2 * np.pi  # Josephson energy
PHI_A = 0.11  # zero-point phase fluctuation, storage mode
PHI_B = 0.204  # zero-point phase fluctuation, buffer mode


def _derived_params(alpha_sq, w_a, kappa_b, E_J, phi_a, phi_b, epsilon_p, n_periods):
    """Couplings, rates and timing derived from the pump — shared by all frames.

    Returns ``(params, T_block)`` where ``params`` is the bookkeeping dict
    stored alongside every sweep result.
    """
    w_b = 2 * w_a

    # couplings derived from the pump
    g = np.sin(epsilon_p) * E_J * phi_a**2 * phi_b
    g2 = jv(1, epsilon_p) * E_J * phi_a**2 * phi_b

    kappa_2 = 4 * g**2 / kappa_b
    kappa_1 = 0.005 * kappa_2

    epsilon_d = 2 * alpha_sq * g2

    T_drive = 2 * jnp.pi / w_a
    T_block = n_periods * T_drive

    params = {
        "g": g,
        "g2": g2,
        "kappa_1": kappa_1,
        "kappa_2": kappa_2,
        "kappa_b": kappa_b,
        "epsilon_d": epsilon_d,
        "w_a": w_a,
        "w_b": w_b,
        "T_drive": T_drive,
    }
    return params, T_block


def _two_mode_operators(n_a, n_b):
    """Annihilation operators of the composite system, ``(a_tot, b_tot)``."""
    a = dq.destroy(n_a)
    b = dq.destroy(n_b)
    Ia = dq.eye(n_a)
    Ib = dq.eye(n_b)

    a_tot = dq.tensor(a, Ib)
    b_tot = dq.tensor(Ia, b)
    return a_tot, b_tot


def build_ats_hamiltonian(
    n_a=20,
    n_b=11,
    alpha_sq=8.5,
    w_a=W_A,
    kappa_b=KAPPA_B,
    E_J=E_J,
    phi_a=PHI_A,
    phi_b=PHI_B,
    epsilon_p=0.1,
    n_periods=1,
):
    """Build the driven full-ATS Lindbladian in the lab frame.

    Mirrors the operator/Hamiltonian cells of ``Chebyshev-Arnoldi.ipynb``.

    Returns
    -------
    Ham : dynamiqs time-dependent operator
        The lab-frame Hamiltonian.
    jump_ops : list
        Lindblad jump operators ``[sqrt(kappa_1) a, sqrt(kappa_b) b]``.
    T_block : float
        Duration of one Floquet block, ``n_periods * 2*pi / w_a``.
    params : dict
        Derived quantities (``g``, ``g2``, ``kappa_1``, ``kappa_2``,
        ``epsilon_d``, ``w_b``, ``T_drive``) for reference.
    """
    params, T_block = _derived_params(
        alpha_sq, w_a, kappa_b, E_J, phi_a, phi_b, epsilon_p, n_periods
    )
    kappa_1 = params["kappa_1"]
    epsilon_d = params["epsilon_d"]
    w_b = params["w_b"]

    a_tot, b_tot = _two_mode_operators(n_a, n_b)

    phi_a_tot = phi_a * (a_tot + dq.dag(a_tot))
    phi_b_tot = phi_b * (b_tot + dq.dag(b_tot))

    non_linear_op = dq.sinm(phi_a_tot + phi_b_tot) - phi_a_tot - phi_b_tot

    H_drive = dq.modulated(
        lambda t: epsilon_d * jnp.cos(w_b * t),
        b_tot + dq.dag(b_tot),
    )

    H_0 = w_a * dq.dag(a_tot) @ a_tot + w_b * dq.dag(b_tot) @ b_tot

    Ham = H_0 + H_drive - 2 * E_J * jnp.sin(epsilon_p) * non_linear_op

    jump_ops = [
        jnp.sqrt(kappa_1) * a_tot,
        jnp.sqrt(kappa_b) * b_tot,
    ]

    return Ham, jump_ops, T_block, params


def build_ats_hamiltonian_rotating(
    n_a=20,
    n_b=11,
    alpha_sq=8.5,
    w_a=W_A,
    kappa_b=KAPPA_B,
    E_J=E_J,
    phi_a=PHI_A,
    phi_b=PHI_B,
    epsilon_p=0.3,
    n_periods=1,
):
    """Build the driven full-ATS Lindbladian in the rotating frame.

    Previously ``build_ats_hamiltonian_r`` in the exact-diagonalization
    script; note its historical default ``epsilon_p=0.3``.

    Returns
    -------
    Ham : dynamiqs time-dependent operator
        The rotating-frame Hamiltonian ``Ham_full_r``.
    jump_ops : list
        Lindblad jump operators ``[sqrt(kappa_1) a, sqrt(kappa_b) b]``.
    T_block : float
        Duration of one Floquet block, ``n_periods * 2*pi / w_a``.
    params : dict
        Derived quantities (``g``, ``g2``, ``kappa_1``, ``kappa_2``,
        ``epsilon_d``, ``w_b``, ``T_drive``) for reference.
    """
    params, T_block = _derived_params(
        alpha_sq, w_a, kappa_b, E_J, phi_a, phi_b, epsilon_p, n_periods
    )
    kappa_1 = params["kappa_1"]
    epsilon_d = params["epsilon_d"]
    w_b = params["w_b"]

    a_tot, b_tot = _two_mode_operators(n_a, n_b)

    a_r = dq.modulated(lambda t: jnp.exp(-1j * w_a * t), a_tot)
    a_dag_r = dq.modulated(lambda t: jnp.exp(1j * w_a * t), dq.dag(a_tot))
    b_r = dq.modulated(lambda t: jnp.exp(-1j * w_b * t), b_tot)
    b_dag_r = dq.modulated(lambda t: jnp.exp(1j * w_b * t), dq.dag(b_tot))

    phi_a_tot_r = phi_a * (a_r + a_dag_r)
    phi_b_tot_r = phi_b * (b_r + b_dag_r)

    non_linear_op_r = dq.sinm(phi_a_tot_r + phi_b_tot_r) - phi_a_tot_r - phi_b_tot_r

    H_drive_r = epsilon_d * (b_tot + dq.dag(b_tot)) / 2

    Ham = -2 * E_J * jnp.sin(epsilon_p) * non_linear_op_r + H_drive_r

    jump_ops = [
        jnp.sqrt(kappa_1) * a_tot,
        jnp.sqrt(kappa_b) * b_tot,
    ]

    return Ham, jump_ops, T_block, params


def build_ats_hamiltonian_interaction(
    n_a=25,
    n_b=11,
    alpha_sq=8.5,
    w_a=W_A,
    kappa_b=KAPPA_B,
    E_J=E_J,
    phi_a=PHI_A,
    phi_b=PHI_B,
    epsilon_p=0.1,
    n_periods=1,
):
    """Build the driven full-ATS Lindbladian in the interaction frame.

    Returns
    -------
    H_I : time-dependent dynamiqs operator
        Driving part of the Hamiltonian.
    jump_ops_I : list
        Time-dependent Lindblad jump operators
        ``[sqrt(kappa_1) a, sqrt(kappa_b) b]``.
    jump_ops_LdL_I : list
        Precomputed ``L^dag L`` operators, for ``dq.mesolve_fast`` (pass as
        ``jump_ops_LdL`` to ``ChebAr``).
    output_phase : jax array
        Per-basis-state phase undoing the frame rotation after one block
        (pass as ``output_phase`` to ``ChebAr``).
    V : numpy array
        Eigenbasis of the static Hamiltonian (frame transformation).
    T_block : float
        Duration of one Floquet block, ``n_periods * 2*pi / w_a``.
    params : dict
        Derived quantities (``g``, ``g2``, ``kappa_1``, ``kappa_2``,
        ``epsilon_d``, ``w_b``, ``T_drive``) for reference.
    """
    params, T_block = _derived_params(
        alpha_sq, w_a, kappa_b, E_J, phi_a, phi_b, epsilon_p, n_periods
    )
    kappa_1 = params["kappa_1"]
    epsilon_d = params["epsilon_d"]
    w_b = params["w_b"]

    a_tot, b_tot = _two_mode_operators(n_a, n_b)

    phi_a_tot = phi_a * (a_tot + dq.dag(a_tot))
    phi_b_tot = phi_b * (b_tot + dq.dag(b_tot))

    non_linear_op = dq.sinm(phi_a_tot + phi_b_tot) - phi_a_tot - phi_b_tot

    H_s = (
        w_a * dq.dag(a_tot) @ a_tot
        + w_b * dq.dag(b_tot) @ b_tot
        - 2 * E_J * jnp.sin(epsilon_p) * non_linear_op
    )

    Hs_mat = H_s.to_numpy()
    Hs_mat = 0.5 * (Hs_mat + Hs_mat.conj().T)  # symmetrize numerically
    lam, V = np.linalg.eigh(Hs_mat)  # H_s = V diag(lam) V^dag
    Vd = V.conj().T

    a_bar = jnp.array(Vd @ a_tot.to_numpy() @ V)  # fixed, built once
    b_bar = jnp.array(Vd @ b_tot.to_numpy() @ V)
    Delta = jnp.array(lam[:, None] - lam[None, :])  # fixed, built once

    def a_tilde_fn(t):
        return dq.asqarray(a_bar * jnp.exp(1j * Delta * t), dims=(n_a, n_b))

    def b_tilde_fn(t):
        return dq.asqarray(b_bar * jnp.exp(1j * Delta * t), dims=(n_a, n_b))

    a_tilde = dq.timecallable(a_tilde_fn)
    b_tilde = dq.timecallable(b_tilde_fn)

    def H_drive_I_fn(t):
        bt = b_tilde_fn(t)
        return epsilon_d * jnp.cos(w_b * t) * (bt + dq.dag(bt))

    H_I = dq.timecallable(H_drive_I_fn)
    jump_ops_I = [jnp.sqrt(kappa_1) * a_tilde, jnp.sqrt(kappa_b) * b_tilde]

    a_dag_a_bar = a_bar.conj().T @ a_bar
    b_dag_b_bar = b_bar.conj().T @ b_bar

    def a_dag_a_tilde_fn(t):
        return dq.asqarray(a_dag_a_bar * jnp.exp(1j * Delta * t), dims=(n_a, n_b))

    def b_dag_b_tilde_fn(t):
        return dq.asqarray(b_dag_b_bar * jnp.exp(1j * Delta * t), dims=(n_a, n_b))

    a_dag_a_tilde = dq.timecallable(a_dag_a_tilde_fn)
    b_dag_b_tilde = dq.timecallable(b_dag_b_tilde_fn)

    jump_ops_LdL_I = [kappa_1 * a_dag_a_tilde, kappa_b * b_dag_b_tilde]

    output_phase = jnp.exp(-1j * jnp.array(lam) * T_block)

    return H_I, jump_ops_I, jump_ops_LdL_I, output_phase, V, T_block, params


def transform_vectorized_state(x_vec, V_from, V_to):
    """Re-express a vectorized operator between two interaction-frame bases.

    ``x_vec`` is a column-major vectorized operator written in the eigenbasis
    ``V_from`` (as returned by :func:`build_ats_hamiltonian_interaction`); the
    result is the same operator written in the eigenbasis ``V_to``. Used to
    warm-start a sweep point from the previous point's Ritz vector when the
    static Hamiltonian — hence its eigenbasis — changes along the sweep
    (e.g. an ``epsilon_p`` sweep; for an ``alpha_sq`` sweep the basis is
    unchanged and this reduces to the identity).
    """
    N = V_from.shape[0]
    rho = np.asarray(x_vec).reshape((N, N), order="F")
    rho_lab = V_from @ rho @ V_from.conj().T
    rho_to = V_to.conj().T @ rho_lab @ V_to
    return jnp.array(rho_to.reshape(-1, order="F"))
