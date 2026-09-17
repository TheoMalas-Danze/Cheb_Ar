"""Exact-diagonalization bit-flip reference for small cats.

**This is a cluster driver.** It runs *on* the anb-compute Ray cluster. Submit
it from anb-dev with ``scripts/cluster/submit.py exact_diagonalization``.

For each ``(eps_p, alpha_sq)`` it builds the full one-period propagator with
``dq.mepropagator`` on the rotating-frame Lindbladian and diagonalizes it
exactly (only tractable for small Hilbert spaces): the bit-flip eigenvalue is
the second largest in magnitude, and the rate is ``-log(mu) / T_block``.

The grid points are completely independent, so they fan out one GPU task each.

Note on size: the eigenvector array is ``(n_eps, n_alpha, (n_a*n_b)**2)``
complex, which at the defaults is already ~130 MB as JSON. ``--no-vectors``
keeps only the eigenvalues, which is usually what you want.
"""

import argparse
import json
import os
import tempfile


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--eps-p", type=float, nargs="+", default=[0.6, 0.8, 1.0],
                   help="pump-strength sweep values")
    p.add_argument("--alpha-sq", type=float, nargs="+",
                   default=[3, 3.5, 4, 4.5, 5, 5.5, 6],
                   help="cat-size sweep values")
    p.add_argument("--n-a", type=int, default=13)
    p.add_argument("--n-b", type=int, default=6)
    p.add_argument("--output", required=True,
                   help="destination for results.json, e.g. "
                        "gs://anb-ray-results/cheb-ar/<run-id>/")
    p.add_argument("--num-cpus", type=int, default=8,
                   help="cpus per task (cap at 28: the driver takes one of 30)")
    p.add_argument("--no-vectors", action="store_true",
                   help="keep only the eigenvalues (the eigenvector array is "
                        "by far the bulk of the output)")
    return p.parse_args()


def _point(eps_p, alpha_sq, n_a, n_b, want_vector):
    """Exact bit-flip eigenpair at one grid point, on a GPU worker."""
    import dynamiqs as dq
    import jax.numpy as jnp
    import numpy as np

    from cheb_ar.models.ats import build_ats_hamiltonian_rotating

    try:
        Ham_full, jump_ops, T_block, _ = build_ats_hamiltonian_rotating(
            n_a=n_a, n_b=n_b, alpha_sq=alpha_sq, epsilon_p=eps_p
        )
        tsave = jnp.array([0.0, T_block])
        propagator = (
            dq.mepropagator(Ham_full, jump_ops, tsave).propagators[-1].to_jax()
        )

        eig_val, eig_vec = jnp.linalg.eig(propagator)
        idx_sort = jnp.argsort(jnp.abs(eig_val))  # sort by magnitude
        bit_flip_mu = eig_val[idx_sort[-2]]  # second largest
        bit_flip_vec = eig_vec[:, idx_sort[-2]]
        bit_flip_lambda = -jnp.log(bit_flip_mu) / T_block

        out = {
            "eps_p": float(eps_p),
            "alpha_sq": float(alpha_sq),
            "mu": complex(np.asarray(bit_flip_mu)),
            "bit_flip_lambda": complex(np.asarray(bit_flip_lambda)),
        }
        if want_vector:
            out["bit_flip_vec"] = np.asarray(bit_flip_vec)
        return out
    except Exception as exc:  # noqa: BLE001 - one bad point must not sink the grid
        from cheb_ar.pipeline import error_entry

        return error_entry(exc, eps_p=float(eps_p), alpha_sq=float(alpha_sq))


def main():
    args = parse_args()

    import numpy as np
    import ray

    from anb_compute import ray as acr
    from cheb_ar.io import to_jsonable

    n_eps, n_alpha = len(args.eps_p), len(args.alpha_sq)
    dim = (args.n_a * args.n_b) ** 2

    refs = {}
    for i, eps_p in enumerate(args.eps_p):
        for j, alpha_sq in enumerate(args.alpha_sq):
            ref = acr.submit(
                _point, float(eps_p), float(alpha_sq),
                args.n_a, args.n_b, not args.no_vectors,
                num_gpus=1, num_cpus=args.num_cpus,
            )
            refs[ref] = (i, j)
    print(f"submitted {len(refs)} task(s) over a {n_eps}x{n_alpha} grid")

    # Same grid layout as the pre-cluster version, so downstream plotting code
    # that indexes [i, j] keeps working. NaN marks a point that never landed.
    lam = np.full((n_eps, n_alpha), np.nan, dtype=complex)
    vec = (
        np.zeros((n_eps, n_alpha, dim), dtype=complex)
        if not args.no_vectors else None
    )
    errors = []

    pending = list(refs)
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        ref = ready[0]
        i, j = refs[ref]
        eps_p, alpha_sq = args.eps_p[i], args.alpha_sq[j]
        try:
            result = ray.get(ref)
        except Exception as exc:  # noqa: BLE001 - infra failure of one task
            print(f"eps_p={eps_p:.2f} alpha_sq={alpha_sq:.1f} LOST: {exc}")
            errors.append({"eps_p": float(eps_p), "alpha_sq": float(alpha_sq),
                           "error": f"{type(exc).__name__}: {exc}"})
            continue
        if "error" in result:
            print(f"eps_p={eps_p:.2f} alpha_sq={alpha_sq:.1f} FAILED: "
                  f"{result['error']}")
            errors.append(result)
            continue
        lam[i, j] = result["bit_flip_lambda"]
        if vec is not None:
            vec[i, j] = result["bit_flip_vec"]
        print(f"eps_p={eps_p:.2f} alpha_sq={alpha_sq:.1f}  "
              f"lambda={result['bit_flip_lambda']:.6e}")

    results = {
        "eps_p_list": to_jsonable(list(args.eps_p)),
        "alpha_sq_list": to_jsonable(list(args.alpha_sq)),
        "n_a": args.n_a,
        "n_b": args.n_b,
        "bit_flip_lambda_array": to_jsonable(lam),
        "errors": errors,
    }
    if vec is not None:
        results["bit_flip_vec_array"] = to_jsonable(vec)

    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "results.json")
        with open(local, "w") as f:
            json.dump(results, f, indent=2)
        uri = args.output if args.output.endswith("/") else args.output + "/"
        acr.save_output(local, uri)
    print(f"saved to {uri}results.json ({len(errors)} failed point(s))")


if __name__ == "__main__":
    main()
