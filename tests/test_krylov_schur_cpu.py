"""CPU checks of the Krylov-Schur solver, off the cluster.

The only part of ``KrylovSchurLindblad`` that needs a GPU is the ``mesolve``
propagator. Everything else -- the two-pass Gram-Schmidt expansion, the thick
restart, target selection, the residual estimate, the resume, the long-block
Rayleigh quotient -- runs anywhere, and is checked here in two scenarios.

**1. Restart policy** (``main_dense``), on a dense non-normal matrix built to
look like the cat-qubit propagator:

* a bulk of eigenvalues filling a disk of radius ``R_BULK`` (Floquet folding
  spreads the bulk in angle, so a disk is the honest model),
* a slow cluster next to 1: the target (real) and a complex pair a factor two
  further from 1, standing in for the phase-flip coherences,
* a few outer bulk modes between the cluster and the disk.

Three restart policies are run at equal basis size. The point of the table is
the claim in ``docs/krylov_schur.md``: keeping the neighbours of the target is
what makes a single-eigenvalue computation converge.

**2. A genuine Lindbladian** (``main_lindblad``): a 12-state master equation
with two weakly connected groups of states, exponentiated exactly. It is trace
preserving and Hermiticity preserving, so -- unlike the dense matrix above --
the traceless subspace really is invariant, as it is for the ATS propagator.
That makes it the test of the parts the dense matrix cannot probe: the trace
projection, the Hermitization of the Ritz vector, and the agreement between the
one-block Ritz value and the ten-block Rayleigh quotient, both against exact
diagonalization.

Run with the workspace pixi python (jax on CPU is enough):

    PYTHONPATH=src JAX_PLATFORMS=cpu python tests/test_krylov_schur_cpu.py
"""

import sys
import time
import types

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# The solver modules call `dynamiqs.set_precision` at import time and nothing
# else until a propagator is built. Without the GPU stack installed (the
# normal case off the cluster), a stub is enough for this test.
try:
    import dynamiqs  # noqa: F401
except ModuleNotFoundError:
    _stub = types.ModuleType("dynamiqs")
    _stub.set_precision = lambda *_args, **_kwargs: None
    sys.modules["dynamiqs"] = _stub

from cheb_ar.solvers.krylov_schur import KrylovSchurLindblad  # noqa: E402

DIM = 400
R_BULK = 0.90
SEED = 0

# Spectrum, in units of one block: target, phase-flip-like pair, outer bulk.
TARGET = 1.0 - 1.0e-4
PAIR = (1.0 - 2.0e-4) * np.exp(1j * np.array([3.0e-4, -3.0e-4]))
OUTER = np.array([0.96, 0.95, 0.94 + 0.02j, 0.94 - 0.02j, 0.93, 0.92])


def build_operator(rng):
    n_bulk = DIM - 1 - len(PAIR) - len(OUTER)
    # bulk uniformly in the disk of radius R_BULK, closed under conjugation
    r = R_BULK * np.sqrt(rng.uniform(size=n_bulk // 2))
    ang = rng.uniform(0, 2 * np.pi, size=n_bulk // 2)
    bulk = r * np.exp(1j * ang)
    bulk = np.concatenate([bulk, bulk.conj()])
    if len(bulk) < n_bulk:
        bulk = np.append(bulk, rng.uniform(-R_BULK, R_BULK))
    lam = np.concatenate([[TARGET], PAIR, OUTER, bulk])
    # mildly non-normal: eigenvector matrix = I + perturbation
    V = np.eye(DIM) + 0.3 * rng.normal(size=(DIM, DIM)) / np.sqrt(DIM)
    A = V @ np.diag(lam) @ np.linalg.inv(V)
    return A, lam


def run_policy(A, lam_target, k, m=30, res_tol=1e-9, max_cycles=60):
    Aj = jnp.asarray(A, dtype=jnp.complex128)
    solver = KrylovSchurLindblad.from_operator(lambda v: Aj @ v, DIM)
    rng = np.random.default_rng(SEED)
    x0 = jnp.asarray(rng.normal(size=DIM) + 1j * rng.normal(size=DIM))
    t0 = time.time()
    out = solver.solve(x0, m=m, k=k, res_tol=res_tol, max_cycles=max_cycles, log=None)
    dt = time.time() - t0
    err = abs(out["mu"] - lam_target)
    # true residual on the dense matrix, independent of the Krylov estimate
    x = np.asarray(out["x_ritz"])
    true_res = np.linalg.norm(A @ x - out["mu"] * x) / abs(out["mu"])
    defects = [h["defect"] for h in out["history"] if h["defect"] is not None]
    return {
        "k": k,
        "converged": out["converged"],
        "cycles": out["n_cycles"],
        "apps": out["n_apply"],
        "eig_err": err,
        "res_est": out["res_rel"],
        "res_true": true_res,
        "max_defect": max(defects) if defects else 0.0,
        "t": dt,
        "out": out,
    }


def main_dense():
    rng = np.random.default_rng(SEED)
    A, lam = build_operator(rng)
    lam_target = lam[np.argmax(lam.real)]
    assert abs(lam_target - TARGET) < 1e-12

    print(f"dense model: dim={DIM}, bulk radius={R_BULK}, target={TARGET:.6f}, "
          f"pair modulus={abs(PAIR[0]):.6f}\n")
    print(f"{'kept k':>6} {'conv':>5} {'cycles':>6} {'apps':>6} {'|mu-lam|':>10} "
          f"{'res est':>10} {'res true':>10} {'max defect':>11} {'time':>6}")
    rows = []
    for k in (1, 3, 8):
        r = run_policy(A, lam_target, k=k)
        rows.append(r)
        print(f"{r['k']:6d} {str(r['converged']):>5} {r['cycles']:6d} {r['apps']:6d} "
              f"{r['eig_err']:10.2e} {r['res_est']:10.2e} {r['res_true']:10.2e} "
              f"{r['max_defect']:11.2e} {r['t']:5.1f}s")

    # --- checks -----------------------------------------------------------
    best = rows[-1]  # k = 8: target + pair + outer bulk kept
    assert best["converged"], "thick restart with k=8 did not converge"
    assert best["eig_err"] < 1e-7, f"eigenvalue error {best['eig_err']:.2e}"
    # the Krylov residual estimate must agree with the true residual
    assert best["res_true"] < 10 * max(best["res_est"], 1e-12), (
        best["res_true"], best["res_est"])
    # the restart must reproduce an (almost) exact Krylov-Schur decomposition
    assert best["max_defect"] < 1e-8, best["max_defect"]

    # Orthonormality of the truncated basis in the returned state.
    Q = np.asarray(best["out"]["state"]["Q"])
    kk = best["out"]["state"]["k"]
    G = Q[: kk + 1].conj() @ Q[: kk + 1].T
    assert np.linalg.norm(G - np.eye(kk + 1)) < 1e-10, "kept basis not orthonormal"

    # Deflating the neighbours must not be slower than the single-vector restart.
    single = rows[0]
    assert (not single["converged"]) or best["apps"] <= single["apps"], (
        single["apps"], best["apps"])

    # --- resume from `state`, as the notebook does after tightening Tsit5 ----
    Aj = jnp.asarray(A, dtype=jnp.complex128)
    solver = KrylovSchurLindblad.from_operator(lambda v: Aj @ v, DIM)
    rng2 = np.random.default_rng(SEED)
    x0 = jnp.asarray(rng2.normal(size=DIM) + 1j * rng2.normal(size=DIM))
    part1 = solver.solve(x0, m=30, k=8, res_tol=1e-4, max_cycles=60, log=None)
    part2 = solver.solve(state=part1["state"], m=30, k=8, res_tol=1e-9,
                         max_cycles=60, log=None)
    assert part2["converged"], "resumed run did not converge"
    assert abs(part2["mu"] - lam_target) < 1e-7, abs(part2["mu"] - lam_target)
    # resuming must not cost more than running straight through
    assert part2["n_apply"] <= 1.3 * best["apps"], (part2["n_apply"], best["apps"])
    print(f"\nresume: {part1['n_apply']} + {part2['n_apply'] - part1['n_apply']} apps, "
          f"|mu - lambda| = {abs(part2['mu'] - lam_target):.2e}")

    # --- long-block Rayleigh quotient, as used for the final rate -----------
    n_blocks = 10
    rq = solver.residual_check(part2["x_ritz"], n_blocks=n_blocks)
    mu_n = complex(rq["mu_RQ"])
    assert abs(mu_n - lam_target**n_blocks) < 1e-8, (mu_n, lam_target**n_blocks)
    # the rate read off the long block must match the one-block rate
    rate_n = complex(rq["rate_RQ"])
    rate_1 = complex(solver.rate_from_mu(lam_target))
    assert abs(rate_n - rate_1) < 1e-8 * abs(rate_1), (rate_n, rate_1)
    print(f"{n_blocks}-block RQ: mu^n = {mu_n.real:.10f}, "
          f"rate {rate_n.real:.6e} vs exact {rate_1.real:.6e}")


# ---------------------------------------------------------------------- #
# 2. A genuine Lindbladian: trace and Hermiticity preserving              #
# ---------------------------------------------------------------------- #
N_A, N_B = 4, 3
N_H = N_A * N_B  # Hilbert dimension; Liouville dimension is N_H**2
T_BLOCK = 1.0


def build_lindblad_propagator(w_in=2.0e-3, w_out=8.0e-6, dephasing=3.0e-4):
    """One-block propagator of a 12-state rate model, as a dense matrix.

    Two groups of six states, fast transitions inside a group and slow ones
    between: the slowest traceless mode is the population imbalance between
    the groups (real eigenvalue, Hermitian -- in fact diagonal -- eigenvector),
    which plays the role of the bit-flip mode. Pure dephasing sets where the
    coherences sit. Column-major vectorization, matching the solver.
    """
    from scipy.linalg import expm

    rng = np.random.default_rng(3)
    half = N_H // 2
    jump_ops = []
    for i in range(N_H):
        for j in range(N_H):
            if i == j:
                continue
            same = (i < half) == (j < half)
            rate = (w_in * (0.5 + rng.uniform())) if same else w_out
            L = np.zeros((N_H, N_H), dtype=complex)
            L[i, j] = np.sqrt(rate)
            jump_ops.append(L)
    jump_ops.append(np.sqrt(dephasing) * np.diag(np.arange(N_H).astype(float)).astype(complex))

    I = np.eye(N_H)
    # column-major vec: vec(A X B) = (B.T kron A) vec(X); no coherent part, so
    # the target stays real (its eigenvector is a population difference).
    L_super = np.zeros((N_H**2, N_H**2), dtype=complex)
    for Lk in jump_ops:
        LdL = Lk.conj().T @ Lk
        L_super += np.kron(Lk.conj(), Lk)
        L_super -= 0.5 * np.kron(I, LdL) + 0.5 * np.kron(LdL.T, I)
    return expm(L_super * T_BLOCK)


def main_lindblad():
    A = build_lindblad_propagator()
    dim = N_H**2
    id_vec = np.eye(N_H).T.reshape(-1)
    e = id_vec / np.linalg.norm(id_vec)

    # exact reference on the traceless subspace
    Pr = np.eye(dim) - np.outer(e, e.conj())
    vals = np.linalg.eigvals(Pr @ A @ Pr)
    vals = vals[np.abs(vals) > 1e-12]  # drop the zero the projection creates
    lam_target = vals[np.argmax(vals.real)]
    trace_err = np.abs(A.conj().T @ id_vec - id_vec).max()

    print(f"\nLindblad stand-in: Hilbert {N_A}x{N_B}, Liouville dim {dim}, "
          f"trace-preservation error {trace_err:.1e}")
    print(f"exact target (largest Re, traceless): {lam_target.real:.12f} "
          f"(Im {lam_target.imag:.1e})")
    slow = np.sort(vals.real)[::-1][:5]
    print("five largest Re:", " ".join(f"{s:.6f}" for s in slow))
    assert trace_err < 1e-12, trace_err
    assert abs(lam_target.imag) < 1e-10, "target must be real (Hermitian eigenvector)"

    Aj = jnp.asarray(A, dtype=jnp.complex128)
    solver = KrylovSchurLindblad.from_operator(lambda v: Aj @ v, dim, T_block=T_BLOCK)
    # give it the density-matrix structure the from_operator path skips, so the
    # trace projection and the Hermitization are exercised
    solver.dims, solver.N = (N_A, N_B), N_H
    solver._build_helpers()
    project = solver.project_traceless
    solver.propagate = jax.jit(lambda v: project(Aj @ project(v)))
    solver._build_expand()

    x0 = solver.make_x0(seed=SEED)
    assert abs(np.vdot(id_vec, np.asarray(x0))) < 1e-10, "seed is not traceless"

    out = solver.solve(x0, m=10, k=4, res_tol=1e-12, max_cycles=40, log=None)
    print(f"solve: converged={out['converged']}, cycles={out['n_cycles']}, "
          f"apps={out['n_apply']}, mu={out['mu'].real:.12f}, "
          f"|mu - exact|={abs(out['mu'] - lam_target):.2e}")
    assert out["converged"]
    assert abs(out["mu"] - lam_target) < 1e-10, abs(out["mu"] - lam_target)

    # The Ritz vector is Hermitized by `solve`; for a real target that must be
    # a no-op up to noise, i.e. the eigenvector really is Hermitian.
    x = np.asarray(out["x_ritz"])
    M = x.reshape(N_H, N_H).T
    assert np.linalg.norm(M - M.conj().T) < 1e-9, "Ritz vector is not Hermitian"
    assert abs(np.trace(M)) < 1e-9, "Ritz vector is not traceless"

    # One block and ten blocks must give the same rate, and the exact one.
    rate_exact = complex(-np.log(lam_target) / T_BLOCK)
    r1 = solver.residual_check(out["x_ritz"], n_blocks=1)
    r10 = solver.residual_check(out["x_ritz"], n_blocks=10)
    rate_1, rate_10 = complex(r1["rate_RQ"]), complex(r10["rate_RQ"])
    print(f"rate: exact {rate_exact.real:.9e} | 1 block {rate_1.real:.9e} "
          f"| 10 blocks {rate_10.real:.9e}")
    print(f"      residual 1 block {float(np.real(r1['res_rel'])):.2e}, "
          f"10 blocks {float(np.real(r10['res_rel'])):.2e}")
    assert abs(rate_1 - rate_exact) < 1e-9 * abs(rate_exact), (rate_1, rate_exact)
    assert abs(rate_10 - rate_exact) < 1e-9 * abs(rate_exact), (rate_10, rate_exact)
    # the ten-block quotient resolves a ten times larger gap, as intended
    assert abs(1 - complex(r10["mu_RQ"])) > 5 * abs(1 - complex(r1["mu_RQ"]))


def main():
    main_dense()
    main_lindblad()
    print("\nall checks passed")


if __name__ == "__main__":
    main()
