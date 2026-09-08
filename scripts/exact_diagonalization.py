"""Exact-diagonalization bit-flip reference for small cats.

Port of ``Cheb_Ar_old/exact_diagonalization/exact_diagonalization_small_cats.py``
to the installable package. Sweeps both ``eps_p`` and ``alpha_sq``; for each
point it builds the full one-period propagator with ``dq.mepropagator`` on the
rotating-frame Lindbladian and diagonalizes it exactly (only tractable for
small Hilbert spaces): the bit-flip eigenvalue is the second largest in
magnitude, and the rate is ``-log(mu) / T_block``.

Example:
    python scripts/exact_diagonalization.py --output results/bit_flip_results.json
"""

import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--eps-p", type=float, nargs="+", default=[0.6, 0.8, 1.0],
                   help="pump-strength sweep values")
    p.add_argument("--alpha-sq", type=float, nargs="+",
                   default=[3, 3.5, 4, 4.5, 5, 5.5, 6],
                   help="cat-size sweep values")
    p.add_argument("--n-a", type=int, default=13)
    p.add_argument("--n-b", type=int, default=6)
    p.add_argument("--output", required=True, help="output JSON path")
    p.add_argument("--gpu-id", default=None,
                   help="sets CUDA_VISIBLE_DEVICES before importing jax")
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # heavy imports only after CUDA_VISIBLE_DEVICES is set
    import dynamiqs as dq
    import jax.numpy as jnp
    import numpy as np

    from cheb_ar.io import to_jsonable
    from cheb_ar.models.ats import build_ats_hamiltonian_rotating

    n_eps = len(args.eps_p)
    n_alpha = len(args.alpha_sq)
    dim = (args.n_a * args.n_b) ** 2

    # (n_eps, n_alpha) for the eigenvalues, (n_eps, n_alpha, dim) for vectors
    bit_flip_lambda_array = np.zeros((n_eps, n_alpha), dtype=complex)
    bit_flip_vec_array = np.zeros((n_eps, n_alpha, dim), dtype=complex)

    for i, eps_p in enumerate(args.eps_p):
        for j, alpha_sq in enumerate(args.alpha_sq):
            print(f"eps_p={eps_p:.2f}  alpha_sq={alpha_sq:.1f}")
            Ham_full, jump_ops, T_block, params = build_ats_hamiltonian_rotating(
                n_a=args.n_a, n_b=args.n_b, alpha_sq=alpha_sq, epsilon_p=eps_p
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

            bit_flip_lambda_array[i, j] = np.array(bit_flip_lambda)
            bit_flip_vec_array[i, j] = np.array(bit_flip_vec)

    results = {
        "eps_p_list": to_jsonable(list(args.eps_p)),
        "alpha_sq_list": to_jsonable(list(args.alpha_sq)),
        "n_a": args.n_a,
        "n_b": args.n_b,
        "bit_flip_lambda_array": to_jsonable(bit_flip_lambda_array),
        "bit_flip_vec_array": to_jsonable(bit_flip_vec_array),
    }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
