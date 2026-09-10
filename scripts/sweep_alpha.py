"""Sweep the cat size ``alpha_sq`` at fixed ``eps_p``, interaction frame.

Port of ``Cheb_Ar_old/loop_alpha_sq/cheb_ar_loop_alpha_sq.py`` to the
installable package and the interaction-frame pipeline
(``build_ats_hamiltonian_interaction`` + ``mesolve_fast`` + ``output_phase``).
Same structure as the old script:

- ``eps_p`` and ``kappa_b`` are fixed across the sweep (the interaction-frame
  eigenbasis does not depend on ``alpha_sq``, so warm vectors need no basis
  rotation).
- Optionally warm-starts each point from the ``x_ritz`` entries of a previous
  results file (``--warm-start-file``, point ``i`` of that file seeds point
  ``i`` of this sweep, like the old script). The vectors are used as-is, so
  the file must come from a run with the same ``n_a``/``n_b``; ideally from
  the same ``eps_p`` (same eigenbasis), otherwise they are merely a rough
  initial guess.
- Failures are caught per point and recorded as an ``"error"`` entry.

Example:
    python scripts/sweep_alpha.py --output results/cheb_ar_vs_alphasq_epsp_1.json
"""

import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--alpha-sq", type=float, nargs="+",
                   default=[3, 4, 5, 6, 7, 8], help="sweep values")
    p.add_argument("--eps-p", type=float, default=1.0)
    p.add_argument("--kappa-b", type=float, default=None,
                   help="fixed kappa_b (default: the model default)")
    p.add_argument("--n-a", type=int, default=20)
    p.add_argument("--n-b", type=int, default=8)
    p.add_argument("--cheb-degree", type=int, default=6)
    p.add_argument("--m-arnoldi-0", type=int, default=60,
                   help="Krylov size of the unfiltered first estimation")
    p.add_argument("--m-arnoldi", type=int, default=80,
                   help="filtered Krylov size")
    p.add_argument("--margin", type=float, default=5e-3)
    p.add_argument("--warm-start-file", default=None,
                   help="results JSON of a previous sweep; its i-th x_ritz "
                        "seeds the i-th point of this sweep")
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
    from cheb_ar.io import load_json, to_jsonable
    from cheb_ar.models.ats import KAPPA_B, build_ats_hamiltonian_interaction

    kappa_b = args.kappa_b if args.kappa_b is not None else KAPPA_B

    def run_for_alphasq(alpha_sq, x0=None):
        """Run the full pipeline for one value of ``alpha_sq``."""
        (
            H_I, jump_ops_I, jump_ops_LdL_I, output_phase, V, T_block, params,
        ) = build_ats_hamiltonian_interaction(
            n_a=args.n_a, n_b=args.n_b, alpha_sq=alpha_sq,
            kappa_b=kappa_b, epsilon_p=args.eps_p,
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
            x0, solver.chebyshev_filter, args.m_arnoldi, warm_start=warm_start
        )

        rate_bf = solver.rate_from_mu(mu_list[-1])
        x_ritz, _ = solver.ritz_vector(Q, H, args.m_arnoldi, target=mu_list[-1])
        res = solver.residual_check(x_ritz)

        return {
            "alpha_sq": alpha_sq,
            "eps_p": args.eps_p,
            "kappa_b": kappa_b,
            "n_a": args.n_a,
            "n_b": args.n_b,
            "cheb_degree": args.cheb_degree,
            "m_arnoldi_0": m_arnoldi_0_used,
            "margin": margin_used,
            "m_arnoldi": args.m_arnoldi,
            "rate_bf": to_jsonable(rate_bf),
            "x_ritz": to_jsonable(x_ritz),
            "res": to_jsonable(res),
            "params": to_jsonable(params),
        }

    x_ritz_warm = None
    if args.warm_start_file is not None:
        warm_data = load_json(args.warm_start_file)
        x_ritz_warm = [d.get("x_ritz") for d in warm_data]

    results = []
    for i, alpha_sq in enumerate(args.alpha_sq):
        x0 = None
        if x_ritz_warm is not None and i < len(x_ritz_warm):
            x0 = x_ritz_warm[i]
        try:
            result = run_for_alphasq(alpha_sq, x0)
        except Exception as exc:
            print(f"alpha_sq = {alpha_sq} FAILED: {exc}")
            results.append({
                "alpha_sq": float(alpha_sq),
                "eps_p": args.eps_p,
                "kappa_b": kappa_b,
                "n_a": args.n_a,
                "n_b": args.n_b,
                "cheb_degree": args.cheb_degree,
                "m_arnoldi_0": args.m_arnoldi_0,
                "m_arnoldi": args.m_arnoldi,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        print(f"alpha_sq = {alpha_sq}")
        print(f"rate_bf  = {result['rate_bf']}")
        print(f"res_rel  = {result['res']['res_rel']}")
        results.append(result)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
