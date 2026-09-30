"""Sweep the pump strength ``eps_p``, interaction frame.

**This is a cluster driver.** It runs *on* the anb-compute Ray cluster. Submit
it from anb-dev with ``scripts/cluster/submit.py sweep_eps_p``.

Each point is solved with Krylov-Schur by default
(:func:`floquet_lindblad.pipeline.solve_point_ks_safe`); ``--solver cheb_ar``
runs the Chebyshev-filtered pipeline instead. The Krylov-Schur defaults are
those of ``notebooks/sweeps/sweep_eps_p.ipynb``: sized for the first point,
which is the hardest (the rate grows with ``eps_p``), and held along the sweep,
with the final block shortening as the gap opens.

Unlike ``sweep_alpha.py``, this sweep is **sequential by construction**: each
point is warm-started from the previous point's Ritz vector, re-expressed in the
new eigenbasis (``transform_vectorized_state``), because the interaction-frame
basis moves with ``eps_p``. Only the first point runs cold, with the larger
Krylov size (``--m`` for Krylov-Schur, ``--m-arnoldi-first`` for ChebAr). That
chain cannot be fanned out without changing the numerics, so the default here
runs one GPU task per point, in order, passing the warm vector along.

``--no-warm-start`` breaks the chain on purpose: every point then runs cold with
the cold-start size, the points become independent, and they are submitted all
at once. More GPU-seconds in total, much less wall clock.

By default ``kappa_b`` is rescaled at each point as ``sin(eps_p)/sin(eps_p_init)
* kappa_b_init`` to keep the adiabatic ratio ``kappa_b / g`` constant;
``--no-keep-adiabatic-ratio`` holds ``kappa_b = kappa_b_init`` fixed instead.
``kappa_b_init`` defaults to the model's ``ats.KAPPA_B``.

A failed point is recorded and the chain restarts cold from the next one.
"""

import argparse
import json
import os
import tempfile


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--eps-p", type=float, nargs="+", default=None,
                   help="sweep values (default: 6 points linspace(0.1, 1.1))")
    p.add_argument("--alpha-sq", type=float, default=3.0)
    p.add_argument("--solver", choices=("krylov_schur", "cheb_ar"),
                   default="krylov_schur")
    p.add_argument("--n-a", type=int, default=None,
                   help="default: 30 (krylov_schur), 20 (cheb_ar)")
    p.add_argument("--n-b", type=int, default=None,
                   help="default: 15 (krylov_schur), 8 (cheb_ar)")
    p.add_argument("--kappa-b-init", type=float, default=None,
                   help="kappa_b at the first sweep point (default: ats.KAPPA_B)")
    p.add_argument("--keep-adiabatic-ratio", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="rescale kappa_b as sin(eps_p) to keep kappa_b / g constant "
                        "(default); --no-keep-adiabatic-ratio keeps kappa_b "
                        "constant at --kappa-b-init")
    p.add_argument("--no-warm-start", action="store_true",
                   help="cold-start every point (all with the cold-start size); "
                        "makes the points independent, so they run in parallel")

    ks = p.add_argument_group("krylov_schur")
    ks.add_argument("--m", type=int, default=50,
                    help="restart size for cold-started points")
    ks.add_argument("--m-warm", type=int, default=36,
                    help="restart size for warm-started points")
    ks.add_argument("--k", type=int, default=12, help="vectors kept per restart")
    ks.add_argument("--max-cycles", type=int, default=35)
    ks.add_argument("--rtol-loop", type=float, default=1e-9)
    ks.add_argument("--atol-loop", type=float, default=1e-10)
    ks.add_argument("--res-tol-loop", type=float, default=1e-8)
    ks.add_argument("--rtol-final", type=float, default=1e-11)
    ks.add_argument("--atol-final", type=float, default=1e-12)
    ks.add_argument("--res-tol", type=float, default=1e-10)
    ks.add_argument("--n-blocks-final", type=int, default=50,
                    help="cap on the final Rayleigh-quotient block")
    ks.add_argument("--rq-resolution", type=float, default=1e-5,
                    help="target 1 - mu**n of the final block, which shortens "
                         "as the gap opens")

    ca = p.add_argument_group("cheb_ar")
    ca.add_argument("--cheb-degree", type=int, default=6)
    ca.add_argument("--m-arnoldi-0", type=int, default=60,
                    help="Krylov size of the unfiltered first estimation")
    ca.add_argument("--m-arnoldi-first", type=int, default=120,
                    help="filtered Krylov size for cold-started points")
    ca.add_argument("--m-arnoldi", type=int, default=80,
                    help="filtered Krylov size for warm-started points")
    ca.add_argument("--margin", type=float, default=5e-3)
    p.add_argument("--output", required=True,
                   help="destination for results.json, e.g. "
                        "gs://anb-ray-results/cheb-ar/<run-id>/")
    p.add_argument("--num-cpus", type=int, default=8,
                   help="cpus per task (cap at 28: the driver takes one of 30)")
    return p.parse_args()


def _strip_chain_state(result):
    """Drop the arrays that only existed to warm-start the next point."""
    return {k: v for k, v in result.items() if k not in ("V",)}


def main():
    args = parse_args()

    import numpy as np
    import ray

    from anb_compute import ray as acr
    from floquet_lindblad.io import to_jsonable
    from floquet_lindblad.models.ats import KAPPA_B
    from floquet_lindblad.pipeline import solve_point_cheb_ar_safe, solve_point_ks_safe

    eps_p_list = (
        list(args.eps_p) if args.eps_p is not None
        else list(np.linspace(0.1, 1.1, 6))
    )
    eps_p_init = eps_p_list[0]

    kappa_b_init = KAPPA_B if args.kappa_b_init is None else args.kappa_b_init

    def kappa_b_for(eps_p):
        if not args.keep_adiabatic_ratio:
            return float(kappa_b_init)
        # keep the adiabatic ratio kappa_b / g constant across the sweep
        return float(np.sin(eps_p) / np.sin(eps_p_init) * kappa_b_init)

    if args.solver == "krylov_schur":
        solve_point_safe = solve_point_ks_safe
        n_a, n_b = args.n_a or 30, args.n_b or 15
        kw = dict(
            k=args.k, max_cycles=args.max_cycles,
            rtol_loop=args.rtol_loop, atol_loop=args.atol_loop,
            res_tol_loop=args.res_tol_loop,
            rtol_final=args.rtol_final, atol_final=args.atol_final,
            res_tol=args.res_tol, n_blocks_final=args.n_blocks_final,
            n_blocks_auto=True, rq_resolution=args.rq_resolution,
        )

        def size_kw(warm):
            return {"m": args.m_warm if warm else args.m}
    else:
        solve_point_safe = solve_point_cheb_ar_safe
        n_a, n_b = args.n_a or 20, args.n_b or 8
        kw = dict(
            cheb_degree=args.cheb_degree,
            m_arnoldi_0=args.m_arnoldi_0,
            margin=args.margin,
        )

        def size_kw(warm):
            return {"m_arnoldi": args.m_arnoldi if warm else args.m_arnoldi_first}
    kw.update(n_a=n_a, n_b=n_b, alpha_sq=args.alpha_sq)
    print(f"solver: {args.solver}  (n_a, n_b) = ({n_a}, {n_b})  "
          f"kappa_b: {'kappa_b / g held' if args.keep_adiabatic_ratio else 'fixed'}, "
          f"{kappa_b_init:.5f} at eps_p = {eps_p_init:g}")
    task_opts = dict(num_gpus=1, num_cpus=args.num_cpus)

    results = []

    if args.no_warm_start:
        # Points are independent: submit them all, consume as they land.
        refs = {}
        for eps_p in eps_p_list:
            ref = acr.submit(
                solve_point_safe,
                eps_p=float(eps_p), kappa_b=kappa_b_for(eps_p),
                **size_kw(warm=False),
                want_x_ritz=True, want_V=False,
                **task_opts, **kw,
            )
            refs[ref] = float(eps_p)
        print(f"submitted {len(refs)} independent task(s) (cold start): "
              f"eps_p = {list(refs.values())}")

        pending = list(refs)
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            ref, eps_p = ready[0], refs[ready[0]]
            try:
                result = ray.get(ref)
            except Exception as exc:  # noqa: BLE001 - infra failure of one task
                print(f"eps_p = {eps_p} LOST: {type(exc).__name__}: {exc}")
                results.append({"eps_p": eps_p, "kappa_b": kappa_b_for(eps_p),
                                "error": f"{type(exc).__name__}: {exc}"})
                continue
            _report(eps_p, result)
            results.append(result)
    else:
        # Sequential chain: point i+1 needs point i's x_ritz *and* its V.
        prev = None
        for eps_p in eps_p_list:
            warm = prev is not None
            ref = acr.submit(
                solve_point_safe,
                eps_p=float(eps_p), kappa_b=kappa_b_for(eps_p),
                **size_kw(warm),
                x0=prev["x_ritz"] if warm else None,
                V_prev=prev["V"] if warm else None,
                # The next point needs this point's eigenbasis to rotate its
                # Ritz vector into, so V has to come back with the result.
                want_x_ritz=True, want_V=True,
                **task_opts, **kw,
            )
            try:
                result = ray.get(ref)
            except Exception as exc:  # noqa: BLE001 - infra failure of one task
                print(f"eps_p = {eps_p} LOST: {type(exc).__name__}: {exc}")
                results.append({"eps_p": float(eps_p),
                                "kappa_b": kappa_b_for(eps_p),
                                "error": f"{type(exc).__name__}: {exc}"})
                prev = None  # next point restarts cold
                continue
            _report(eps_p, result)
            results.append(_strip_chain_state(result))
            # A failed point breaks the chain: restart cold from the next one.
            prev = result if "error" not in result else None

    results.sort(key=lambda d: d["eps_p"])
    for r in results:
        r.setdefault("solver", args.solver)  # which pipeline wrote it
        r.setdefault("keep_adiabatic_ratio", args.keep_adiabatic_ratio)

    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "results.json")
        with open(local, "w") as f:
            json.dump(to_jsonable(results), f, indent=2)
        uri = args.output if args.output.endswith("/") else args.output + "/"
        acr.save_output(local, uri)
    print(f"saved {len(results)} point(s) to {uri}results.json")


def _report(eps_p, result):
    if "error" in result:
        print(f"eps_p = {eps_p} FAILED: {result['error']}")
    else:
        print(f"eps_p = {eps_p}  rate_bf = {result['rate_bf']}  "
              f"res_rel = {_res_rel(result)}  "
              f"({result['timings_s']['total']:.0f} s)")


def _res_rel(result):
    """The final relative residual, wherever the solver's pipeline puts it."""
    return result["res_rel_n"] if "res_rel_n" in result else result["res"]["res_rel"]


if __name__ == "__main__":
    main()
