# Plain Arnoldi bit-flip rate solver

Documentation for `src/cheb_ar/solvers/arnoldi_no_cheb.py` (the
`ArnoldiLindblad` solver) and the notebook built on it,
`tests/test_arnoldi_no_cheb.ipynb`. Like everything else here it is GPU-only
(JAX + `dynamiqs`) and cannot be executed on a CPU-only machine. The ATS model
builders are documented in `docs/models_ats.md`, the filtered solver in
`docs/cheb_ar.md`.

## 1. What this is for

`ArnoldiLindblad` computes the same quantity as `ChebAr` — the decay rate of the
leading non-trivial eigenvalue of the trace-projected one-block propagator `P` —
by running **unfiltered** Arnoldi to convergence. No ellipse fit, no Chebyshev
polynomial, no escalation ladder.

That makes it slower than `ChebAr` for the same accuracy, which is the point: it
is the independent reference that the filtered result is checked against. If the
two disagree, the Chebyshev setup (`margin`, `cheb_degree`, the ellipse fit) is
the suspect, not the physics. It is also the fallback when the ellipse fit
misbehaves on a sweep point.

Both solvers take the same interaction-frame hooks from
`build_ats_hamiltonian_interaction`, so they are drop-in comparable:
`jump_ops_LdL` (switches the propagator to `dq.mesolve_fast`) and
`output_phase`.

## 2. Conventions

- **Column-major vectorization**, matching `dq.vectorize`:
  `vec(M) = M.T.reshape(-1)` and `unvec(v) = v.reshape(N, N).T`. Rebuild density
  matrices with `solver.unvec`, **not** a bare `reshape(N, N)` — the latter
  returns the transpose, i.e. the complex conjugate for Hermitian input.
- Arnoldi basis `Q` has shape `(m+1, dim)`, row `k` being the `k`-th basis
  vector; `H` has shape `(m+1, m)`.
- All Krylov vectors are Hermitian and traceless (Hermitian seed, and `P`
  preserves both), so `H` is real up to integrator noise. `np.abs(H.imag).max()`
  is therefore a free estimate of the noise floor — the notebook returns it as
  `H_imag_max` and draws it on the convergence plot, since `res_rel` cannot
  meaningfully go below it.

## 3. API

| Member | Purpose |
|---|---|
| `ArnoldiLindblad(Ham, jump_ops, T_block, ...)` | `jump_ops_LdL`, `output_phase`, `dims`, `rtol`/`atol`, `require_gpu` as for `ChebAr` |
| `make_x0(seed=0)` | random Hermitian traceless seed (hermitized Ginibre) |
| `propagate(rho_vec)` | jitted one-block propagator, trace-projected |
| `arnoldi(x0, m_new, *, m_old, Q_old, H_old, mu_ref, stride)` | run or **extend** a factorization; returns `(Q, H, steps, mu_list)` |
| `track_target(H, m_new, ...)` | Ritz value nearest `mu_ref` at each sampled step |
| `ritz_vector(Q, H, m, target)` | Ritz vector nearest `target`; returns `(x_ritz, rho_ritz)` |
| `residual_check(x_ritz)` | `mu_RQ`, `res_rel`, `rate_RQ` on the true `P` |
| `rate_from_mu(mu)` | `-log(mu) / T_block` |

Two things to get right:

- `arnoldi` returns **four** values. `steps` and `mu_list` line up elementwise,
  so plot against `steps`, not `range(m_new)` — with `stride > 1` they differ.
- The `m` passed to `ritz_vector` must be the step count actually reached
  (`steps[-1]`). A smaller value silently discards Krylov vectors and yields a
  worse eigenvector than `mu_list[-1]` suggests. The pre-cluster notebook did
  exactly this: it factorized to 360 steps and then asked for the Ritz vector at
  60.

**Warm restart.** `arnoldi` extends an existing factorization in place when given
`m_old`, `Q_old` and `H_old`, reusing the earlier matvecs:

```python
Q, H, steps, mu_list = solver.arnoldi(x0, 60)
Q, H, steps, mu_list = solver.arnoldi(x0, 100, m_old=60, Q_old=Q, H_old=H)
```

## 4. Running it on the cluster

`tests/test_arnoldi_no_cheb.ipynb` mirrors `tests/test_cheb_ar.ipynb`: the whole
pipeline is one picklable `run_arnoldi` function handed to `acr.run` with
`num_gpus=1`, returning **plain numpy/python objects only** (the client has
neither jax nor dynamiqs). Wigner grids are computed on the worker; the client
only plots.

The bootstrap/auth cell is copied verbatim from `test_cheb_ar.ipynb` — keep the
two in sync, and see `RAY_SETUP_NOTES.md` for the two separate auth steps (the
Ray token, and the GCS credentials needed to fetch the result envelope).

**Do not return `Q` from the worker.** At `n_a=40, n_b=15` it is
`(m+1, N**2) = (91, 360000)` complex128, roughly 260 MB, and it has to be
pickled through GCS. `run_arnoldi` returns `H` (kilobytes) and `x_ritz`
(about 5.8 MB) instead. This is the main practical difference from the
pre-cluster notebooks, which simply kept everything in the kernel.

Defaults are `n_a=40, n_b=15, alpha_sq=10, epsilon_p=0.1, n_periods=6,
m_arnoldi=90`, carried over from `arnoldi_no_cheb/test_ar_no_cheb.ipynb`.
Physics constants are **not** arguments — they come from
`cheb_ar.models.ats`, which is their single source of truth. Note that
`n_a=40, n_b=15` is a considerably larger space than the `ChebAr` notebook's
`(25, 11)`; see the Fock-truncation discussion in `tests/test_cheb_ar.ipynb`
for why that is the safer choice at `alpha_sq=10`.
