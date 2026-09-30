"""Sweep the pump strength ``eps_p``, interaction frame.

**This is a cluster driver.** It runs *on* the anb-compute Ray cluster. Submit
it from anb-dev with ``scripts/cluster/submit.py sweep_eps_p``.

Unlike ``sweep_alpha.py``, this sweep is **sequential by construction**: each
point is warm-started from the previous point's Ritz vector, re-expressed in the
new eigenbasis (``transform_vectorized_state``), because the interaction-frame
basis moves with ``eps_p``. Only the first point runs cold, with the larger
``--m-arnoldi-first``. That chain cannot be fanned out without changing the
numerics, so the default here runs one GPU task per point, in order, passing the
warm vector along.

``--no-warm-start`` breaks the chain on purpose: every point then runs cold with
``--m-arnoldi-first``, the points become independent, and they are submitted all
at once. More GPU-seconds in total, much less wall clock.

``kappa_b`` is rescaled at each point as ``sin(eps_p)/sin(eps_p_init) *
kappa_b_init`` to keep the adiabatic ratio ``kappa_b / g`` constant.

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
                   help="cold-start every point (all with --m-arnoldi-first); "
                        "makes the points independent, so they run in parallel")
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
    from floquet_lindblad.pipeline import solve_point_safe

    eps_p_list = (
        list(args.eps_p) if args.eps_p is not None
        else list(np.linspace(0.1, 1.1, 6))
    )
    eps_p_init = eps_p_list[0]

    def kappa_b_for(eps_p):
        # keep the adiabatic ratio kappa_b / g constant across the sweep
        return float(np.sin(eps_p) / np.sin(eps_p_init) * args.kappa_b_init)

    kw = dict(
        n_a=args.n_a,
        n_b=args.n_b,
        alpha_sq=args.alpha_sq,
        cheb_degree=args.cheb_degree,
        m_arnoldi_0=args.m_arnoldi_0,
        margin=args.margin,
    )
    task_opts = dict(num_gpus=1, num_cpus=args.num_cpus)

    results = []

    if args.no_warm_start:
        # Points are independent: submit them all, consume as they land.
        refs = {}
        for eps_p in eps_p_list:
            ref = acr.submit(
                solve_point_safe,
                eps_p=float(eps_p), kappa_b=kappa_b_for(eps_p),
                m_arnoldi=args.m_arnoldi_first,
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
            m_arnoldi = args.m_arnoldi if warm else args.m_arnoldi_first
            ref = acr.submit(
                solve_point_safe,
                eps_p=float(eps_p), kappa_b=kappa_b_for(eps_p),
                m_arnoldi=m_arnoldi,
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
              f"res_rel = {result['res']['res_rel']}  "
              f"({result['timings_s']['total']:.0f} s)")


if __name__ == "__main__":
    main()
