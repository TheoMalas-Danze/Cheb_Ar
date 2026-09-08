"""Sweep the pump strength ``eps_p``, interaction frame.

Port of ``Cheb_Ar_old/loop_eps_p/cheb_ar_loop_eps_p.py`` to the installable
package and the interaction-frame pipeline (``build_ats_hamiltonian_interaction``
+ ``mesolve_fast`` + ``output_phase``). Same structure as the old script:

- ``kappa_b`` is rescaled at each point as ``sin(eps_p)/sin(eps_p_init) *
  kappa_b_init`` to keep the adiabatic ratio ``kappa_b / g`` constant.
- The first point runs cold with ``--m-arnoldi-first`` Krylov vectors; each
  subsequent point is warm-started from the previous point's Ritz vector with
  the smaller ``--m-arnoldi``. New here: the warm vector is re-expressed in
  the new point's eigenbasis (``transform_vectorized_state``), since the
  interaction-frame basis changes with ``eps_p``.
- Failures are caught per point and recorded as an ``"error"`` entry; the
  next point then restarts cold (this also fixes the old first-point
  ``NameError`` bug).

Example:
    python scripts/sweep_eps_p.py --output results/cheb_ar_vs_epsp_alphasq_6.json
"""

import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--eps-p", type=float, nargs="+", default=None,
                   help="sweep values (default: 6 points linspace(0.1, 1.1))")
    p.add_argument("--alpha-sq", type=float, default=6.0)
    p.add_argument("--n-a", type=int, default=20)
    p.add_argument("--n-b", type=int, default=8)
    p.add_argument("--cheb-degree", type=int, default=6)
    p.add_argument("--m-arnoldi-0", type=int, default=60,
                   help="Krylov size of the unfiltered first estimation")
    p.add_argument("--m-arnoldi-first", type=int, default=120,
                   help="filtered Krylov size for cold-started points")
    p.add_argument("--m-arnoldi", type=int, default=80,
                   help="filtered Krylov size for warm-started points")
    p.add_argument("--margin", type=float, default=5e-3)
    p.add_argument("--kappa-b-init", type=float, default=0.6 / 10.4,
                   help="kappa_b at the first sweep point")
    p.add_argument("--no-warm-start", action="store_true",
                   help="cold-start every point (all with --m-arnoldi-first)")
    p.add_argument("--output", required=True, help="output JSON path")
    p.add_argument("--gpu-id", default=None,
                   help="sets CUDA_VISIBLE_DEVICES before importing jax")
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # heavy imports only after CUDA_VISIBLE_DEVICES is set
    import numpy as np

    from cheb_ar import ChebAr
    from cheb_ar.io import to_jsonable
    from cheb_ar.models.ats import (
        build_ats_hamiltonian_interaction,
        transform_vectorized_state,
    )

    def run_for_epsp(eps_p, kappa_b, m_arnoldi, x0=None, V_prev=None):
        """Run the full pipeline for one value of ``eps_p``.

        Returns ``(result_raw, result_json)``; ``result_raw`` keeps the jax
        objects needed to warm-start the next point in-process.
        """
        (
            H_I, jump_ops_I, jump_ops_LdL_I, output_phase, V, T_block, params,
        ) = build_ats_hamiltonian_interaction(
            n_a=args.n_a, n_b=args.n_b, alpha_sq=args.alpha_sq,
            kappa_b=kappa_b, epsilon_p=eps_p,
        )

        solver = ChebAr(
            H_I, jump_ops_I, T_block,
            jump_ops_LdL=jump_ops_LdL_I, output_phase=output_phase,
            dims=(args.n_a, args.n_b), cheb_degree=args.cheb_degree,
        )

        if x0 is None:
            x0 = solver.make_x0(seed=0)
            warm_start = False
        else:
            # previous point's Ritz vector, re-expressed in this eigenbasis
            x0 = transform_vectorized_state(x0, V_prev, V)
            warm_start = True

        # First estimation + ellipse fit, with an escalation ladder on
        # failure: rerun the estimation with m_arnoldi_0 * sqrt(2), then * 2,
        # then keep that estimation and halve the margin down to 1e-5.
        min_margin = 1e-5
        m_schedule = [
            args.m_arnoldi_0,
            int(round(args.m_arnoldi_0 * np.sqrt(2))),
            2 * args.m_arnoldi_0,
        ]
        m_arnoldi_0_used = None
        margin_used = None
        ritz_vals = None
        last_exc = None
        for m0 in m_schedule:
            try:
                _, _, ritz_vals = solver.first_estimation(x0, m_arnoldi=m0)
                solver.setup_chebyshev(ritz_vals, margin=args.margin)
                m_arnoldi_0_used, margin_used = m0, args.margin
                break
            except Exception as exc:
                last_exc = exc
                print(
                    f"setup_chebyshev failed "
                    f"(m_arnoldi_0={m0}, margin={args.margin:.3e}): {exc}"
                )
        if margin_used is None:
            if ritz_vals is None:
                raise last_exc  # even the first estimation itself failed
            margin = args.margin / 2
            while margin >= min_margin:
                try:
                    solver.setup_chebyshev(ritz_vals, margin=margin)
                    m_arnoldi_0_used, margin_used = m_schedule[-1], margin
                    break
                except Exception as exc:
                    last_exc = exc
                    print(
                        f"setup_chebyshev failed "
                        f"(m_arnoldi_0={m_schedule[-1]}, margin={margin:.3e}): {exc}"
                    )
                    margin /= 2
            if margin_used is None:
                raise last_exc

        Q, H, mu_list = solver.arnoldi_hessenberg(
            x0, solver.chebyshev_filter, m_arnoldi, warm_start=warm_start
        )

        rate_bf = solver.rate_from_mu(mu_list[-1])
        x_ritz, _ = solver.ritz_vector(Q, H, m_arnoldi, target=mu_list[-1])
        res = solver.residual_check(x_ritz)

        result_raw = {"x_ritz": x_ritz, "V": V}
        result_json = {
            "eps_p": eps_p,
            "alpha_sq": args.alpha_sq,
            "kappa_b": kappa_b,
            "n_a": args.n_a,
            "n_b": args.n_b,
            "cheb_degree": args.cheb_degree,
            "m_arnoldi_0": m_arnoldi_0_used,
            "margin": margin_used,
            "m_arnoldi": m_arnoldi,
            "rate_bf": to_jsonable(rate_bf),
            "x_ritz": to_jsonable(x_ritz),
            "res": to_jsonable(res),
            "params": to_jsonable(params),
        }
        return result_raw, result_json

    eps_p_list = args.eps_p if args.eps_p is not None else np.linspace(0.1, 1.1, 6)
    eps_p_init = eps_p_list[0]

    results = []
    prev = None  # raw result of the last successful point
    for eps_p in eps_p_list:
        # keep the adiabatic ratio kappa_b / g constant across the sweep
        kappa_b = np.sin(eps_p) / np.sin(eps_p_init) * args.kappa_b_init

        warm = prev is not None and not args.no_warm_start
        m_arnoldi = args.m_arnoldi if warm else args.m_arnoldi_first
        try:
            if warm:
                result_raw, result = run_for_epsp(
                    eps_p, kappa_b, m_arnoldi,
                    x0=prev["x_ritz"], V_prev=prev["V"],
                )
            else:
                result_raw, result = run_for_epsp(eps_p, kappa_b, m_arnoldi)
        except Exception as exc:
            print(f"eps_p = {eps_p} FAILED: {exc}")
            results.append({
                "eps_p": float(eps_p),
                "alpha_sq": args.alpha_sq,
                "kappa_b": float(kappa_b),
                "n_a": args.n_a,
                "n_b": args.n_b,
                "cheb_degree": args.cheb_degree,
                "m_arnoldi_0": args.m_arnoldi_0,
                "m_arnoldi": m_arnoldi,
                "error": f"{type(exc).__name__}: {exc}",
            })
            prev = None
            continue

        print(f"eps_p = {eps_p}")
        print(f"rate_bf  = {result['rate_bf']}")
        print(f"res_rel  = {result['res']['res_rel']}")
        results.append(result)
        prev = result_raw

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
