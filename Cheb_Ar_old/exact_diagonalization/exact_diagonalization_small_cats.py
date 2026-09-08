import jax
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import jax.numpy as jnp
wanted_type_complex = jnp.complex128
wanted_type_real = jnp.float64
print("Devices:", jax.devices())
if not any(d.platform == "gpu" for d in jax.devices()):
    raise RuntimeError("JAX is not using a GPU.")
import sys
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
from scipy.special import jv
jax.config.update("jax_enable_x64", True)
import sys
sys.path.insert(0, "/home/tmalasda/dynamiqs")
import dynamiqs as dq
dq.set_precision("double")
assert (
    dq.__file__ == "/home/tmalasda/dynamiqs/dynamiqs/__init__.py"
), "Unexpected dynamiqs install; check sys.path."
import json

n_a = 13
n_b = 6

def build_ats_hamiltonian_r(
    n_a=20,
    n_b=11,
    alpha_sq=8.5,
    w_a=25.338776456203686,
    kappa_b=5 / 10.4,
    E_J=37 * 2 * np.pi,
    phi_a=0.11,
    phi_b=0.204,
    epsilon_p=.3,
    n_periods=1,
):
    """Build the driven full-ATS Lindbladian in the rotating frame.

    Mirrors the operator/Hamiltonian cells of ``Chebyshev-Arnoldi.ipynb``.

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
    w_b = 2 * w_a

    # couplings derived from the pump
    g = np.sin(epsilon_p) * E_J * phi_a**2 * phi_b
    g2 = jv(1, epsilon_p) * E_J * phi_a**2 * phi_b

    kappa_2 = 4 * g**2 / kappa_b
    kappa_1 = 0.005 * kappa_2

    epsilon_d = 2 * alpha_sq * g2

    # Operators
    a = dq.destroy(n_a)
    b = dq.destroy(n_b)
    Ia = dq.eye(n_a)
    Ib = dq.eye(n_b)

    a_tot = dq.tensor(a, Ib)
    b_tot = dq.tensor(Ia, b)

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
    return Ham, jump_ops, T_block, params

def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (jax.Array, np.ndarray)):
        return _to_jsonable(np.asarray(obj).tolist())
    if isinstance(obj, (np.complexfloating, complex)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


eps_p_list   = [.6, .8, 1]
alpha_sq_list = [3, 3.5, 4, 4.5, 5, 5.5, 6]

n_eps   = len(eps_p_list)
n_alpha = len(alpha_sq_list)
dim     = (n_a * n_b)**2

# Pre-allocate: shape (n_eps, n_alpha) for scalars, (n_eps, n_alpha, dim) for vectors
bit_flip_lambda_array = np.zeros((n_eps, n_alpha),      dtype=complex)
bit_flip_vec_array    = np.zeros((n_eps, n_alpha, dim),  dtype=complex)

for i, eps_p in enumerate(eps_p_list):
    for j, alpha_sq in enumerate(alpha_sq_list):
        print(f"eps_p={eps_p:.2f}  alpha_sq={alpha_sq:.1f}")
        Ham_full, jump_ops, T_block, params = build_ats_hamiltonian_r(
            n_a=n_a, n_b=n_b, alpha_sq=alpha_sq, epsilon_p=eps_p
        )
        tsave      = jnp.array([0.0, T_block])
        propagator = dq.mepropagator(Ham_full, jump_ops, tsave).propagators[-1].to_jax()

        eig_val, eig_vec = jnp.linalg.eig(propagator)
        idx_sort  = jnp.argsort(jnp.abs(eig_val))          # sort by magnitude
        bit_flip_mu     = eig_val[idx_sort[-2]]             # second largest
        bit_flip_vec    = eig_vec[:, idx_sort[-2]]
        bit_flip_lambda = -jnp.log(bit_flip_mu) / T_block

        bit_flip_lambda_array[i, j] = np.array(bit_flip_lambda)
        bit_flip_vec_array[i, j]    = np.array(bit_flip_vec)

# --- Save to JSON ---
results = {
    "eps_p_list":            _to_jsonable(eps_p_list),
    "alpha_sq_list":         _to_jsonable(alpha_sq_list),
    "bit_flip_lambda_array": _to_jsonable(bit_flip_lambda_array),   # (n_eps, n_alpha)
    "bit_flip_vec_array":    _to_jsonable(bit_flip_vec_array),       # (n_eps, n_alpha, dim)
}

output_path = "bit_flip_results_eps_p_min_0_4.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {output_path}")
        
