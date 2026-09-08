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
sys.path.insert(0, "/home/tmalasda/dev")
from cheb_ar import ChebAr
import numpy as np
import dynamiqs as dq
from scipy.special import jv
import json

n_a = 20
n_b = 8
cheb_degree = 6
m_arnoldi_0 = 60
m_arnoldi_first = 120
m_arnoldi_generic = 80
margin = 5e-3
alpha_sq = 6

kappa_b_init = .6 / 10.4
eps_p_init = .1
E_J=37 * 2 * np.pi
phi_a=0.11
phi_b=0.204
g_init = np.sin(eps_p_init) * E_J * phi_a**2 * phi_b
adiabatic_constant = kappa_b_init / g_init
print(adiabatic_constant)

OUTPUT_PATH = "/home/tmalasda/output/Cheb_Ar/21_07_2026/cheb_ar_vs_epsp_alphasq_6.json"

def build_ats_hamiltonian(
    n_a=20,
    n_b=11,
    alpha_sq=8.5,
    w_a=25.338776456203686,
    kappa_b=5 / 10.4,
    E_J=37 * 2 * np.pi,
    phi_a=0.11,
    phi_b=0.204,
    epsilon_p=0.1,
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

    phi_a_tot = phi_a*(a_tot + dq.dag(a_tot))
    phi_b_tot = phi_b*(b_tot + dq.dag(b_tot))

    non_linear_op = dq.sinm(phi_a_tot + phi_b_tot) - phi_a_tot - phi_b_tot

    H_drive = dq.modulated(
        lambda t: epsilon_d * jnp.cos(w_b*t),
        b_tot + dq.dag(b_tot)
        )

    H_0 = w_a*a_tot@dq.dag(a_tot) + w_b*b_tot@dq.dag(b_tot)

    Ham = H_0 + H_drive - 2*E_J*jnp.sin(epsilon_p)*non_linear_op 

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
    """Recursively convert jax/numpy arrays and complex numbers to JSON types.

    Complex values become ``{"real": ..., "imag": ...}``; arrays become nested
    lists; numpy/jax scalars become plain Python ``int``/``float``.
    """
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    # jax / numpy arrays -> numpy, then recurse on the python object
    if isinstance(obj, (jax.Array, np.ndarray)):
        return _to_jsonable(np.asarray(obj).tolist())
    if isinstance(obj, (np.complexfloating, complex)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj

def run_for_epsp(eps_p, kappa_b, m_arnoldi, x0 = None):
    """Run the full Chebyshev-Arnoldi pipeline for one value of ``eps_p``.

    Returns a JSON-serializable dict with ``rate_bf``, ``x_ritz`` and ``res``.
    """
    Ham_full_r, jump_ops, T_block, params = build_ats_hamiltonian(
        n_a=n_a, n_b=n_b, epsilon_p=eps_p, alpha_sq=alpha_sq, kappa_b=kappa_b
    )

    solver = ChebAr(
        Ham_full_r, jump_ops, n_a=n_a, n_b=n_b, T_block=T_block, cheb_degree=cheb_degree
    )

    if x0 is  None: 
        x0 = solver.make_x0(seed=0)
        warm_start = False
    else : warm_start = True
    _, _, ritz_vals = solver.first_estimation(x0, m_arnoldi=m_arnoldi_0)

    solver.setup_chebyshev(ritz_vals, margin=margin)

    Q, H, mu_list = solver.arnoldi_hessenberg(x0, solver.chebyshev_filter, m_arnoldi, warm_start = warm_start)

    rate_bf = solver.rate_from_mu(mu_list[-1])
    x_ritz, _ = solver.ritz_vector(Q, H, m_arnoldi)
    res = solver.residual_check(x_ritz)

    result_raw = {
        "eps_p": eps_p,
        "n_a": n_a,
        "n_b": n_b,
        "cheb_degree": cheb_degree,
        "m_arnoldi_0": m_arnoldi_0,
        "m_arnoldi": m_arnoldi,
        "rate_bf": rate_bf,
        "x_ritz": x_ritz,
        "res": res,
        "params": params,
    }
    result_json = {
        "eps_p": eps_p,
        "n_a": n_a,
        "n_b": n_b,
        "cheb_degree": cheb_degree,
        "m_arnoldi_0": m_arnoldi_0,
        "m_arnoldi": m_arnoldi,
        "rate_bf": _to_jsonable(rate_bf),
        "x_ritz": _to_jsonable(x_ritz),
        "res": _to_jsonable(res),
        "params": _to_jsonable(params),
    }

    return result_raw, result_json


if __name__ == "__main__":
    eps_p_list = np.linspace(0.1, 1.1, 6)
    results_raw = []
    results = []

    try:
        result_raw, result = run_for_epsp(eps_p_list[0], kappa_b_init, m_arnoldi_first, x0 = None)

    except Exception as exc:
        print(f"eps_p = {eps_p_list[0]} FAILED: {exc}")
        results.append({
            "eps_p": eps_p_list[0],
            "n_a": n_a,
            "n_b": n_b,
            "cheb_degree": cheb_degree,
            "m_arnoldi_0": m_arnoldi_0,
            "m_arnoldi": m_arnoldi_first,
            "error": f"{type(exc).__name__}: {exc}",
        })
    print(f"eps_p = {eps_p_list[0]}")
    print(f"rate_bf  = {result['rate_bf']}")
    print(f"res_rel  = {result['res']['res_rel']}")
    results.append(result)
    results_raw.append(result_raw)

    for (idx_eps,eps_p) in enumerate(eps_p_list[1:]):
        kappa_b = np.sin(eps_p)/np.sin(eps_p_init)*kappa_b_init
        try:
            x_ritz_warm = results_raw[idx_eps]['x_ritz']
            result_raw, result  = run_for_epsp(eps_p, kappa_b, m_arnoldi_generic, x0 = x_ritz_warm)
        except Exception as exc:
            print(f"eps_p = {eps_p} FAILED: {exc}")
            results.append({
                "eps_p": eps_p,
                "n_a": n_a,
                "n_b": n_b,
                "cheb_degree": cheb_degree,
                "m_arnoldi_0": m_arnoldi_0,
                "m_arnoldi": m_arnoldi_generic,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        print(f"eps_p = {eps_p}")
        print(f"rate_bf  = {result['rate_bf']}")
        print(f"res_rel  = {result['res']['res_rel']}")

        results.append(result)
        results_raw.append(result_raw)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {OUTPUT_PATH}")
