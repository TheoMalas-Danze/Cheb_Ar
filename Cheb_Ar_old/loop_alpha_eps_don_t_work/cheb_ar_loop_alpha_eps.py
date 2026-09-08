import jax
import os
import jax.numpy as jnp
wanted_type_complex = jnp.complex128
wanted_type_real    = jnp.float64
print("Devices:", jax.devices())
if not any(d.platform == "gpu" for d in jax.devices()):
    raise RuntimeError("JAX is not using a GPU.")
import numpy as np
from scipy.special import jv
import json
import multiprocessing as mp

# --- Global hyperparameters ---
n_a         = 20
n_b         = 8
cheb_degree = 6
m_arnoldi_0 = 60
m_arnoldi   = 10
margin      = 5e-3

OUTPUT_DIR  = "/home/tmalasda/output/Cheb_Ar/10_07_2026"

# ---------- helpers (unchanged) ----------
def _to_jsonable(obj):
    if isinstance(obj, dict):                        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):               return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (jax.Array, np.ndarray)):     return _to_jsonable(np.asarray(obj).tolist())
    if isinstance(obj, (np.complexfloating, complex)): return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, (np.floating, float)):        return float(obj)
    if isinstance(obj, (np.integer, int)):           return int(obj)
    return obj


def _from_jsonable(obj):
    if isinstance(obj, dict):
        if set(obj.keys()) == {"real", "imag"}:
            return complex(obj["real"], obj["imag"])
        return {k: _from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        converted = [_from_jsonable(v) for v in obj]
        if all(isinstance(v, (int, float, complex, np.ndarray)) for v in converted):
            return np.array(converted)
        return converted
    return obj


def load_json(path):
    with open(path, "r") as f:
        return _from_jsonable(json.load(f))


# ---------- per-worker function: full alpha_sq loop for one eps_p ----------

def run_alphasq_loop(eps_p, x_ritz_warm):
    """Run the full alpha_sq sweep for one eps_p. Called in a separate process."""
    import sys
    sys.path.insert(0, "/home/tmalasda/dev")
    from cheb_ar import ChebAr
    import dynamiqs as dq
    alpha_sq_list = [3]#, 4, 5, 6, 7, 8]
    results = []

    def build_ats_hamiltonian(
        n_a=20, n_b=11, alpha_sq=8.5,
        w_a=25.338776456203686, kappa_b=5/10.4,
        E_J=37*2*np.pi, phi_a=0.11, phi_b=0.204,
        epsilon_p=0.1, n_periods=1,
    ):
        w_b       = 2 * w_a
        g         = np.sin(epsilon_p) * E_J * phi_a**2 * phi_b
        g2        = jv(1, epsilon_p)  * E_J * phi_a**2 * phi_b
        kappa_2   = 4 * g**2 / kappa_b
        kappa_1   = 0.005 * kappa_2
        epsilon_d = 2 * alpha_sq * g2

        a  = dq.destroy(n_a);  b  = dq.destroy(n_b)
        Ia = dq.eye(n_a);      Ib = dq.eye(n_b)
        a_tot = dq.tensor(a, Ib);  b_tot = dq.tensor(Ia, b)

        phi_a_tot     = phi_a * (a_tot + dq.dag(a_tot))
        phi_b_tot     = phi_b * (b_tot + dq.dag(b_tot))
        non_linear_op = dq.sinm(phi_a_tot + phi_b_tot) - phi_a_tot - phi_b_tot

        H_drive = dq.modulated(lambda t: epsilon_d * jnp.cos(w_b*t), b_tot + dq.dag(b_tot))
        H_0     = w_a*a_tot@dq.dag(a_tot) + w_b*b_tot@dq.dag(b_tot)
        Ham     = H_0 + H_drive - 2*E_J*jnp.sin(epsilon_p)*non_linear_op

        jump_ops = [jnp.sqrt(kappa_1)*a_tot, jnp.sqrt(kappa_b)*b_tot]
        T_drive  = 2*jnp.pi / w_a
        T_block  = n_periods * T_drive
        params   = dict(g=g, g2=g2, kappa_1=kappa_1, kappa_2=kappa_2, kappa_b=kappa_b,
                        epsilon_d=epsilon_d, w_a=w_a, w_b=w_b, T_drive=T_drive)
        return Ham, jump_ops, T_block, params
    

    for (idx_alpha, alpha_sq) in enumerate(alpha_sq_list):
        x0 = x_ritz_warm[idx_alpha]
        try:
            Ham_full_r, jump_ops, T_block, params = build_ats_hamiltonian(
                n_a=n_a, n_b=n_b, epsilon_p=eps_p, alpha_sq=alpha_sq
            )
            solver = ChebAr(
                Ham_full_r, jump_ops, n_a=n_a, n_b=n_b,
                T_block=T_block, cheb_degree=cheb_degree
            )


            _, _, ritz_vals = solver.first_estimation(x0, m_arnoldi=m_arnoldi_0)
            solver.setup_chebyshev(ritz_vals, margin=margin)

            Q, H, mu_list = solver.arnoldi_hessenberg(x0, solver.chebyshev_filter, m_arnoldi)
            rate_bf       = solver.rate_from_mu(mu_list[-1])
            x_ritz, _     = solver.ritz_vector(Q, H, m_arnoldi, target=mu_list[-1])
            res           = solver.residual_check(x_ritz)

            print(f"[eps_p={eps_p}  alpha_sq={alpha_sq}]  rate_bf={_to_jsonable(rate_bf)}  res_rel={res['res_rel']}")

            results.append({
                "eps_p": eps_p, "alpha_sq": alpha_sq,
                "n_a": n_a, "n_b": n_b,
                "cheb_degree": cheb_degree, "m_arnoldi_0": m_arnoldi_0, "m_arnoldi": m_arnoldi,
                "rate_bf": _to_jsonable(rate_bf),
                "x_ritz":  _to_jsonable(x_ritz),
                "res":     _to_jsonable(res),
                "params":  _to_jsonable(params),
            })


        except Exception as exc:
            print(f"[eps_p={eps_p}  alpha_sq={alpha_sq}] FAILED: {exc}")
            results.append({
                "eps_p": eps_p, "alpha_sq": alpha_sq,
                "error": f"{type(exc).__name__}: {exc}",
            })
            # don't update x0 — keep last good vector

    # Each worker saves its own file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"cheb_ar_epsp_{str(eps_p).replace('.','_')}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eps_p={eps_p}] Saved -> {out_path}")

    return results


# ---------- main ----------

if __name__ == "__main__":
    data_eps_2 = load_json("/home/tmalasda/output/Cheb_Ar/07_07_2026/cheb_ar_vs_alphaqs_epsp_.2.json")
    x_ritz_warm  = np.array([d["x_ritz"] for d in data_eps_2])
    eps_p_list = [0.4, 0.6, 0.8, 1.0]   # one per core

    mp.set_start_method("spawn", force=True)

    with mp.Pool(processes=4) as pool:
        all_results = pool.starmap(
            run_alphasq_loop,
            zip(eps_p_list, x_ritz_warm)   # each worker gets (eps_p, x0)
        )

    # Optional combined summary
    flat_results = [r for sublist in all_results for r in sublist]
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(flat_results, f, indent=2)
    print(f"\nAll done. Summary saved to {summary_path}")