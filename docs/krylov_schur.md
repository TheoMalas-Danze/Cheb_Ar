# Thick-restarted Arnoldi (Krylov–Schur) bit-flip rate solver

Documentation for `src/floquet_lindblad/solvers/krylov_schur.py` (the
`KrylovSchurLindblad` solver), the cluster notebook
`notebooks/solvers/krylov_schur.ipynb` and the CPU test `tests/test_krylov_schur_cpu.py`.
The propagator is GPU-only (JAX + `dynamiqs`), like the other two solvers; the
restart algebra itself runs anywhere, which is what the CPU test exercises.

## 1. What this is for

Same target as `ArnoldiLindblad` and `ChebAr`: the eigenvalue of largest real
part of the trace-projected one-block propagator `P`, converted to a rate. The
difference is the iteration:

- it runs on **`P` itself** (one drive period, `T_block = 2π/ω_a`), not on
  `P^n` or on a polynomial `p(P)`. Any polynomial Krylov space is a subspace of
  the Krylov space of `P` with the same number of period integrations, so
  nothing is gained by the polynomial except a smaller basis — which the
  restart provides instead;
- it **restarts** when the basis reaches `m` vectors, truncating to the `k`
  Ritz vectors closest to the unit circle (the target plus its neighbours) and
  continuing Arnoldi from the residual direction. Memory is bounded at `m+1`
  vectors and orthogonalisation at `O(m)` GEMVs per step, whatever the
  total number of applications;
- it stops on the **residual norm** of the target Ritz vector. The Ritz value
  is a poor stopping test here: `1 - mu` per period is `2.5e-6` at
  `alpha_sq = 3` and smaller for larger cats, i.e. at the level of the
  integrator tolerance;
- the **rate** comes from a Rayleigh quotient of the converged eigenvector on
  a long block (`P^10`) at tight integrator tolerance, which resolves
  `1 - mu^10` well above the noise. The iteration itself can run at a relaxed
  tolerance (see §4).

## 2. Why keep neighbours when only one eigenvalue is wanted

Restarting with `k` Ritz vectors is mathematically the implicit restart with
exact shifts at the `m - k` discarded Ritz values (Morgan 1996, Stewart 2001):
the polynomial in `P` one would otherwise build "from the Ritz values of the
previous cycle" is applied for free. Convergence of the target then proceeds
as if the eigenvalues of the other kept vectors were absent from the spectrum,
and this holds while they are still approximate. Two consequences:

- the **phase-flip pair** must be kept. At small `alpha_sq` it sits within a
  factor of a few of the target's distance to 1; a single-vector restart
  (`k = 1`) has to re-separate the target from it every cycle and stalls;
- keeping the **outermost bulk modes** lowers the effective bulk radius that
  sets the number of applications per digit.

The CPU test makes this concrete on a dense model matrix (`dim = 400`, bulk in
a disk of radius 0.9, target `1 - 1e-4`, a complex pair at `1 - 2e-4`, six
outer bulk modes, `m = 30`, `res_tol = 1e-9`):

| kept `k` | converged | cycles | applications | eigenvalue error |
|---|---|---|---|---|
| 1 | no (60-cycle budget) | 60 | 1741 | 8.5e-05 (locked between target and pair) |
| 3 | yes | 16 | 435 | 2.9e-12 |
| 8 | yes | 16 | 360 | 3.0e-13 |

## 3. Algorithm

State: a Krylov–Schur decomposition of order `j`,
`P Q[:j].T = Q[:j].T B[:j, :j] + Q[j] B[j, :j]`, with `Q` of shape `(m+1, dim)`
(row `i` = basis vector `i`, unfilled rows exactly zero) and `B` of shape
`(m+1, m)`.

1. **Expand** (`_expand`, jitted, static `k`, `m`): Arnoldi steps from order
   `k` to `m`. Each step applies `P` once and orthogonalises with two passes of
   classical Gram–Schmidt, each one GEMV against the whole basis (the zero rows
   contribute nothing, so no masking). Then the trace projection, the norm,
   and the new column of `B`.
2. **Ritz pairs** (`ritz_pairs`, host): eigenpairs `(theta_i, z_i)` of
   `B[:m, :m]`; the exact residual norm of Ritz vector `y_i = Q[:m].T z_i` is
   `|B[m, :] @ z_i|` — free.
3. **Target** (`select_target`): largest real part (`"LR"`, default), largest
   modulus, or nearest a given complex number.
4. **Restart** (`restart`, host + one device GEMM): keep the target and the
   `k - 1` other Ritz values of largest modulus; orthonormalise their
   eigenvectors into `W` (QR); the new basis is `W.T @ Q[:m]` plus the old
   residual vector `Q[m]`; the new projected block is `S = W^H B W` and the
   coupling row `B[m, :] @ W`. Because the kept eigenvectors span an invariant
   subspace of `B[:m, :m]`, this is an exact Krylov–Schur decomposition up to a
   `defect` that is recorded per cycle (`~1e-15` in the CPU test).
5. Repeat until `res_rel <= res_tol` or `max_cycles`.

After the first restart the Krylov vectors are complex combinations (a kept
complex pair is not Hermitian), so `B` is no longer real up to noise as in
`ArnoldiLindblad`; the imaginary part of the target Ritz value is the noise
indicator instead, and the returned Ritz vector is Hermitized (right for the
real target).

## 4. API

| Member | Purpose |
|---|---|
| `KrylovSchurLindblad(Ham, jump_ops, T_block, *, jump_ops_LdL, output_phase, dims, rtol, atol, require_gpu)` | same constructor as `ArnoldiLindblad`; `rtol`/`atol` are the **loop** tolerances |
| `KrylovSchurLindblad.from_operator(apply, dim)` | wrap an arbitrary linear map (no Lindbladian, no projection); used by the CPU test |
| `make_x0(seed)` | random Hermitian traceless seed |
| `propagate(v)` | jitted one-block propagator at the current tolerance |
| `set_tolerance(rtol, atol)` | switch the loop's integrator tolerance (propagators cached per pair, compiled once each) |
| `solve(x0=None, *, m=40, k=10, res_tol=1e-8, max_cycles=20, target="LR", state=None, log=print)` | run the restarted iteration; returns `mu`, `res_rel`, `x_ritz`, `converged`, `n_apply`, `n_cycles`, `history`, `ritz`, `state` |
| `ritz_pairs(B, m)`, `select_target(theta, target)`, `restart(...)` | the host-side pieces, exposed for inspection |
| `ritz_vector(Q, B, m, target)` | Ritz vector of an order-`m` decomposition (parity with `ArnoldiLindblad`) |
| `residual_check(x, n_blocks=1, rtol=None, atol=None)` | Rayleigh quotient, relative residual and rate on `P**n_blocks`, optionally at a tighter tolerance |
| `rate_from_mu(mu, n_blocks=1)` | `-log(mu) / (n_blocks * T_block)` |

`solve` never returns `Q` to the caller except inside `state`, which is meant
to be passed back to `solve` on the same worker; do not ship `state` across
the cluster wire (`Q` is `(m+1) * N**2` complex numbers).

**Tolerance schedule.** The intended use, and what the notebook does:

```python
solver = KrylovSchurLindblad(..., rtol=1e-7, atol=1e-8)         # loop
out = solver.solve(x0, m=40, k=10, res_tol=1e-6)                 # relaxed cycles
solver.set_tolerance(1e-9, 1e-10)
out = solver.solve(state=out["state"], m=40, k=10, res_tol=1e-8) # polish
final = solver.residual_check(out["x_ritz"], n_blocks=10, rtol=1e-9, atol=1e-10)
rate = final["rate_RQ"]
```

A relaxed tolerance shows up as a floor in the residual curve, not as a wrong
answer, because the residual is what is monitored. Resuming from `state` after
tightening keeps the subspace; the residual re-adjusts within the first cycle.
That floor is also why the notebook caps the relaxed phase with its own cycle
budget (`max_cycles_loop`, half of `max_cycles` by default) instead of waiting
for `res_tol_loop`, which may sit below what the relaxed integrator can reach.

**Choosing `m` and `k`.** `m = 40`, `k = 10` is the starting point. `k` must
cover the target plus the phase-flip pair plus whatever outer bulk modes the
Ritz spectrum of the first cycles shows between them and the dense bulk
(`history[i]["theta_kept"]` lists what was kept). `m - k` is the number of
applications per cycle; too small and the bulk is barely damped per cycle,
too large and memory grows for no gain.

## 5. Running it on the cluster

`notebooks/solvers/krylov_schur.ipynb` mirrors `notebooks/solvers/arnoldi_no_cheb.ipynb`:
the same bootstrap/auth cell (keep them in sync), one picklable
`run_krylov_schur` handed to `acr.run` with `num_gpus=1`, plain numpy back.
It runs `alpha_sq = 3` on `(n_a, n_b) = (25, 11)` twice — loop at
`1e-7/1e-8` then polish at `1e-9/1e-10`, and tight throughout — and compares
the 10-block Rayleigh-quotient rate with `reports/benchmark_methods.md`
(`arnoldi`, 6-period block, `m = 90`: `9.910757e-06` after 540 period
integrations). Per-cycle Ritz values, residuals and the kept set come back
in `history`; the basis does not.

`notebooks/sweeps/sweep_alpha.ipynb` is the sweep form of the same
thing: the structure of `notebooks/sweeps/legacy_cheb_ar/sweep_alpha.ipynb` (one job, one GPU task
per `alpha_sq`, the perturbative comparison at the end) with this solver in
place of the `ChebAr` pipeline. There is no `pipeline.solve_point` for
`KrylovSchurLindblad`, so the per-point work is a notebook-level
`solve_point_ks`, shipped by value to each task; it returns the rates, the
residuals, the cost and the two Fock marginals, never the basis or the Ritz
vector. Truncation and tolerances follow a ladder in `alpha_sq`
(`(n_a, n_b)` = 20/10, 25/12, 35/16, 38/19 at `alpha_sq` < 2, 3.5, 6.5, and
above), with
`rtol_final` tightening from `1e-9` to `1e-12`, `res_tol` one decade above it,
and `n_blocks_final` growing 10 → 100 — because `1 - mu` per period *is* the
rate, ~2.5e-6 at `alpha_sq = 3` and falling like `exp(-2 alpha_sq)`, so the long
block is what keeps it resolvable above the integrator noise.

`notebooks/sweeps/sweep_eps_p.ipynb` is the same thing for the `eps_p`
sweep, i.e. the structure of `notebooks/sweeps/legacy_cheb_ar/sweep_eps_p.ipynb`: sequential by
default, each point warm-started from the previous one's Ritz vector rotated
into the new eigenbasis (`ats.transform_vectorized_state` — both solvers
vectorize column-major, so it applies to `KrylovSchurLindblad` unchanged),
with `m` trimmed for warm points but `k` left alone, since what `k` has to
cover is a property of the spectrum and not of the starting vector.
`WARM_START = False` reverts to the cold, parallel control run. Tolerances are
*fixed* along this sweep, unlike the `alpha_sq` one: `1 - mu` is the rate and
the rate grows with `eps_p`, so the first point is the hardest and one setting
sized for it (band 3 of the ladder above) covers the rest. The adaptation runs
the other way — `n_blocks_final` is a cap, and the final block is shortened to
whatever gives `1 - mu**n ~ rq_resolution` (1e-5 by default) as the gap opens.

## 6. CPU test

```
PYTHONPATH=src JAX_PLATFORMS=cpu python tests/test_krylov_schur_cpu.py
```

Two scenarios, neither needing a GPU. It stubs `dynamiqs` when the GPU stack
is not installed (the solver modules only call `set_precision` at import).

**Restart policy**, on the dense model matrix of §2: runs `k = 1, 3, 8` at
`m = 30`, prints the table above, and asserts convergence at `k = 8`,
eigenvalue error `< 1e-7`, the Krylov residual estimate matching the true
residual, restart defect `< 1e-8`, an orthonormal kept basis, and that
deflation is not slower than the single-vector restart. It also resumes a
half-converged run from its `state` (what the notebook does after tightening
the integrator) and checks the ten-block Rayleigh quotient against the exact
eigenvalue.

**A genuine Lindbladian**: a 12-state rate model with two weakly connected
groups of states, exponentiated exactly (`scipy.linalg.expm`). Unlike a dense
random matrix it is trace preserving and Hermiticity preserving, so the
traceless subspace is really invariant, as for the ATS propagator. The exact
target is `0.999904004608`, the next modes sit at `0.9907`; the solver reaches
it in 3 cycles and 22 applications at `m = 10`, `k = 4`. Asserted: eigenvalue
to `< 1e-10` of the dense reference, a Ritz vector that is Hermitian and
traceless (so `hermitize_ritz` is a no-op on the real target, as assumed),
and one-block and ten-block Rayleigh-quotient rates both matching the exact
rate to `1e-9` relative.
