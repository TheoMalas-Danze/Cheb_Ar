"""Sweep the cat size ``alpha_sq`` at fixed ``eps_p``, interaction frame.

Each point is solved with Krylov-Schur by default
(:func:`floquet_lindblad.pipeline.solve_point_ks_safe`); ``--solver cheb_ar``
runs the Chebyshev-filtered pipeline instead. The Krylov-Schur defaults are
band 3 of the calibrated ladder in ``notebooks/sweeps/sweep_alpha.ipynb``
(``alpha_sq < 6.5``), used for every point: past that, or to save GPU time at
small cats, pass the band's settings explicitly.

**This is a cluster driver.** It runs *on* the anb-compute Ray cluster and fans
the sweep out, one GPU task per ``alpha_sq``. Submit it from anb-dev with
``scripts/cluster/submit.py sweep_alpha``; running it on a laptop does nothing
useful, since it has no cluster connection.

The points are mutually independent here — ``eps_p`` and ``kappa_b`` are fixed,
so the interaction-frame eigenbasis does not move and no point needs the
previous one's answer. That is exactly the shape Ray wants: one job, N tasks,
the autoscaler sizing the pool. (Contrast ``sweep_eps_p.py``, whose warm-start
chain is inherently sequential.)

Warm starts, if any, come from a previous run's results file (point ``i`` seeds
point ``i``), read from ``gs://`` or a local path. The vectors are used as-is,
so the file must come from a run with the same ``n_a``/``n_b``, ideally the same
``eps_p``.

Results are written to a GCS prefix: the job's own disk is destroyed when the
job ends, so a local ``--output`` path would simply vanish.
"""

import argparse
import json
import os
import tempfile


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--alpha-sq", type=float, nargs="+",
                   default=[1.5, 2, 2.5, 3, 3.5, 4], help="sweep values (|alpha|^2)")
    p.add_argument("--eps-p", type=float, default=1.0)
    p.add_argument("--kappa-b", type=float, default=None,
                   help="fixed kappa_b (default: the model default)")
    p.add_argument("--solver", choices=("krylov_schur", "cheb_ar"),
                   default="krylov_schur")
    p.add_argument("--n-a", type=int, default=None,
                   help="default: 35 (krylov_schur), 20 (cheb_ar)")
    p.add_argument("--n-b", type=int, default=None,
                   help="default: 16 (krylov_schur), 8 (cheb_ar)")

    ks = p.add_argument_group("krylov_schur")
    ks.add_argument("--m", type=int, default=50, help="restart size")
    ks.add_argument("--k", type=int, default=12, help="vectors kept per restart")
    ks.add_argument("--max-cycles", type=int, default=35)
    ks.add_argument("--rtol-loop", type=float, default=1e-9)
    ks.add_argument("--atol-loop", type=float, default=1e-10)
    ks.add_argument("--res-tol-loop", type=float, default=1e-8)
    ks.add_argument("--rtol-final", type=float, default=1e-11)
    ks.add_argument("--atol-final", type=float, default=1e-12)
    ks.add_argument("--res-tol", type=float, default=1e-10)
    ks.add_argument("--n-blocks-final", type=int, default=50,
                    help="length of the final Rayleigh-quotient block")

    ca = p.add_argument_group("cheb_ar")
    ca.add_argument("--cheb-degree", type=int, default=6)
    ca.add_argument("--m-arnoldi-0", type=int, default=60,
                    help="Krylov size of the unfiltered first estimation")
    ca.add_argument("--m-arnoldi", type=int, default=80,
                    help="filtered Krylov size")
    ca.add_argument("--margin", type=float, default=5e-3)
    p.add_argument("--warm-start-file", default=None,
                   help="results JSON of a previous sweep (gs:// or local); "
                        "its i-th x_ritz seeds the i-th point of this sweep")
    p.add_argument("--output", required=True,
                   help="destination for results.json, e.g. "
                        "gs://anb-ray-results/cheb-ar/<run-id>/")
    p.add_argument("--num-cpus", type=int, default=8,
                   help="cpus per task (cap at 28: the driver takes one of 30)")
    p.add_argument("--no-x-ritz", action="store_true",
                   help="omit x_ritz from the results (much smaller output, "
                        "but the run cannot then seed a warm start)")
    return p.parse_args()


def main():
    args = parse_args()

    from anb_compute import ray as acr
    from floquet_lindblad.io import from_jsonable, to_jsonable
    from floquet_lindblad.pipeline import solve_point_cheb_ar_safe, solve_point_ks_safe

    if args.solver == "krylov_schur":
        solve_point_safe = solve_point_ks_safe
        n_a, n_b = args.n_a or 35, args.n_b or 16
        kw = dict(
            m=args.m, k=args.k, max_cycles=args.max_cycles,
            rtol_loop=args.rtol_loop, atol_loop=args.atol_loop,
            res_tol_loop=args.res_tol_loop,
            rtol_final=args.rtol_final, atol_final=args.atol_final,
            res_tol=args.res_tol, n_blocks_final=args.n_blocks_final,
        )
    else:
        solve_point_safe = solve_point_cheb_ar_safe
        n_a, n_b = args.n_a or 20, args.n_b or 8
        kw = dict(
            cheb_degree=args.cheb_degree,
            m_arnoldi_0=args.m_arnoldi_0,
            m_arnoldi=args.m_arnoldi,
            margin=args.margin,
        )
    kw.update(
        n_a=n_a,
        n_b=n_b,
        eps_p=args.eps_p,
        kappa_b=args.kappa_b,
        want_x_ritz=not args.no_x_ritz,
    )
    print(f"solver: {args.solver}  (n_a, n_b) = ({n_a}, {n_b})")

    # Optional warm start: point i of the previous run seeds point i of this one.
    x_ritz_warm = []
    if args.warm_start_file is not None:
        raw = acr.load_output(args.warm_start_file, loader=json.load, mode="r")
        x_ritz_warm = [from_jsonable(d).get("x_ritz") for d in raw]
        print(f"warm-start file: {len(x_ritz_warm)} vector(s)")

    # One task per point. Resources are per task, so there is no worker count to
    # choose: the autoscaler grows the gpu pool for whatever is pending.
    refs = {}
    for i, alpha_sq in enumerate(args.alpha_sq):
        x0 = x_ritz_warm[i] if i < len(x_ritz_warm) else None
        ref = acr.submit(solve_point_safe, alpha_sq=float(alpha_sq), x0=x0,
                         num_gpus=1, num_cpus=args.num_cpus, **kw)
        refs[ref] = float(alpha_sq)
    print(f"submitted {len(refs)} task(s): alpha_sq = {list(refs.values())}")

    # Consume as they land, so a slow point does not hide the finished ones.
    # Deliberately ray.wait/ray.get rather than acr.as_completed: as_completed
    # calls ray.get inside a generator, so a task that dies for an *infra*
    # reason (node preempted, OOM, retries exhausted) would raise out of the
    # loop and discard every point already collected. `solve_point_safe`
    # handles its own numerical failures; this handles the rest.
    import ray

    results = []
    pending = list(refs)
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        ref = ready[0]
        alpha_sq = refs[ref]
        try:
            result = ray.get(ref)
        except Exception as exc:  # noqa: BLE001 - infra failure of one task
            print(f"alpha_sq = {alpha_sq} LOST: {type(exc).__name__}: {exc}")
            results.append({
                "alpha_sq": alpha_sq, "eps_p": args.eps_p, "n_a": n_a,
                "n_b": n_b, "solver": args.solver,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        if "error" in result:
            print(f"alpha_sq = {alpha_sq} FAILED: {result['error']}")
        else:
            print(f"alpha_sq = {alpha_sq}  rate_bf = {result['rate_bf']}  "
                  f"res_rel = {_res_rel(result)}  "
                  f"({result['timings_s']['total']:.0f} s)")
        results.append(result)

    results.sort(key=lambda d: d["alpha_sq"])
    for r in results:
        r.setdefault("solver", args.solver)  # which pipeline wrote it

    # The job's disk dies with the job, so stage locally then upload.
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "results.json")
        with open(local, "w") as f:
            json.dump(to_jsonable(results), f, indent=2)
        uri = args.output if args.output.endswith("/") else args.output + "/"
        acr.save_output(local, uri)
    print(f"saved {len(results)} point(s) to {uri}results.json")


def _res_rel(result):
    """The final relative residual, wherever the solver's pipeline puts it."""
    return result["res_rel_n"] if "res_rel_n" in result else result["res"]["res_rel"]


if __name__ == "__main__":
    main()
