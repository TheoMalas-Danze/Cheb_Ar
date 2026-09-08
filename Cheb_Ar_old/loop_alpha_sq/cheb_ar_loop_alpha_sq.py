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
import matplotlib.pyplot as plt
import json

n_a = 20
n_b = 8
cheb_degree = 6
m_arnoldi_0 = 60
m_arnoldi = 80
margin = 5e-3
eps_p = 1

OUTPUT_PATH = "/home/tmalasda/output/Cheb_Ar/10_07_2026/cheb_ar_vs_alphaqs_epsp_.8.json"

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

def _from_jsonable(obj):
    """Recursively convert JSON types back to numpy/complex objects.
    
    ``{"real": ..., "imag": ...}`` dicts become complex numbers;
    lists become numpy arrays if they contain numeric/complex values.
    """
    if isinstance(obj, dict):
        # Complex number encoding
        if set(obj.keys()) == {"real", "imag"}:
            return complex(obj["real"], obj["imag"])
        # Regular dict: recurse on values
        return {k: _from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        converted = [_from_jsonable(v) for v in obj]
        # Convert to numpy array if all elements are numeric or complex
        if all(isinstance(v, (int, float, complex, np.ndarray)) for v in converted):
            return np.array(converted)
        return converted
    return obj  # int, float, str → keep as-is


def load_json(path):
    """Load a JSON file and return a dict with numpy/complex values."""
    with open(path, "r") as f:
        raw = json.load(f)
    return _from_jsonable(raw)

def run_for_alphasq(alpha_sq, x0 = None):
    """Run the full Chebyshev-Arnoldi pipeline for one value of ``eps_p``.

    Returns a JSON-serializable dict with ``rate_bf``, ``x_ritz`` and ``res``.
    """
    Ham_full_r, jump_ops, T_block, params = build_ats_hamiltonian(
        n_a=n_a, n_b=n_b, epsilon_p=eps_p, alpha_sq=alpha_sq
    )

    solver = ChebAr(
        Ham_full_r, jump_ops, n_a=n_a, n_b=n_b, T_block=T_block, cheb_degree=cheb_degree
    )

    if x0 is  None: x0 = solver.make_x0(seed=0)

    _, _, ritz_vals = solver.first_estimation(x0, m_arnoldi=m_arnoldi_0)

    solver.setup_chebyshev(ritz_vals, margin=margin)

    Q, H, mu_list = solver.arnoldi_hessenberg(x0, solver.chebyshev_filter, m_arnoldi)

    rate_bf = solver.rate_from_mu(mu_list[-1])
    x_ritz, _ = solver.ritz_vector(Q, H, m_arnoldi, target=mu_list[-1])
    res = solver.residual_check(x_ritz)

    return {
        "alpha_sq": alpha_sq,
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


if __name__ == "__main__":
    alpha_sq_list = [3,4,5,6,7,8]
    data_eps_2 = load_json("/home/tmalasda/output/Cheb_Ar/07_07_2026/cheb_ar_vs_alphaqs_epsp_.2.json")
    x_ritz_warm  = np.array([d["x_ritz"] for d in data_eps_2])
    results = []

    for (i,alpha_sq) in enumerate(alpha_sq_list):
        try:
            result = run_for_alphasq(alpha_sq, x_ritz_warm[i])
        except Exception as exc:
            print(f"alpha_sq = {alpha_sq} FAILED: {exc}")
            results.append({
                "alpha_sq": alpha_sq,
                "n_a": n_a,
                "n_b": n_b,
                "cheb_degree": cheb_degree,
                "m_arnoldi_0": m_arnoldi_0,
                "m_arnoldi": m_arnoldi,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        print(f"alpha_sq = {alpha_sq}")
        print(f"rate_bf  = {result['rate_bf']}")
        print(f"res_rel  = {result['res']['res_rel']}")

        results.append(result)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {OUTPUT_PATH}")
