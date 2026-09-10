# Chebyshev–Arnoldi bit-flip rate solver

Documentation for `src/cheb_ar/solvers/cheb_ar.py` (the `ChebAr` solver) and
the sweep scripts built on top of it (`scripts/sweep_eps_p.py`,
`scripts/sweep_alpha.py`, `scripts/exact_diagonalization.py`). All of it
originates from the pre-merge notebook `Chebyshev-Arnoldi.ipynb` (kept
untracked in `Archive/`) and is GPU-only (JAX + `dynamiqs`); it cannot be
executed on a CPU-only machine. The ATS model builders are documented in
`docs/models_ats.md`, the JSON helpers in `docs/io.md`.

## 1. What this code computes

Given a driven, dissipative full-ATS (Asymmetrically Threaded SQUID) system
described by a Floquet Lindbladian, the code estimates the **bit-flip rate**:
the decay rate of the leading non-trivial eigenvalue of the one-period,
trace-projected propagator `P` of the master equation. This eigenvalue sits
just inside the unit disk (`|mu| ≲ 1`); the small gap to 1 is precisely the
protection rate of interest for a bosonic-cat-style memory.

Because `P` is only available as a black-box action on a vectorized density
matrix (one call = one call to `dynamiqs.mesolve` over a full drive period),
the dominant eigenpair is extracted with a **Chebyshev-filtered, warm-startable
Arnoldi iteration**:

1. **`first_estimation`** — a short, unfiltered Arnoldi run on `P` gives a
   rough Ritz spectrum.
2. **`setup_chebyshev`** — the bit-flip Ritz value is set aside as the target;
   an ellipse enclosing the rest of the spectrum is fit, and a scalar
   Chebyshev polynomial `p` optimal for that ellipse is built. `p(P)` damps
   every eigenvalue except the target, so it can be recovered cheaply and
   accurately.
3. **`arnoldi_hessenberg`** — Arnoldi iteration is re-run, but on the filtered
   operator `p(P)` instead of `P` directly. At every step, the current Ritz
   value of `p(P)` is inverted through the Chebyshev map to recover the
   corresponding eigenvalue `mu` of the real operator `P` (`recover_mu_list` /
   `_invert_chebyshev`), tracked by continuity so it doesn't jump to a
   different branch as the subspace grows. This step is **warm-restartable**:
   an existing `(Q, H)` factorization can be extended from `m_old` to `m_new`
   Krylov vectors without recomputation.
4. **`ritz_vector`** reconstructs the full Ritz vector / density matrix for
   the selected eigenvalue, and **`residual_check`** verifies it against the
   true (unfiltered) propagator via a Rayleigh quotient and relative residual.
5. **`rate_from_mu`** converts the eigenvalue to a rate via
   `-log(mu) / T_block`.

`scipy_dominant_eig` is a slower CPU/`scipy.sparse.linalg.eigs` reference
path used to cross-check the JAX pipeline on small Hilbert spaces.

## 2. `src/cheb_ar/solvers/cheb_ar.py` — the `ChebAr` class

Model-agnostic: it only needs a Hamiltonian and jump operators (the Liouvillian
data), and works purely on the vectorized/projected propagator. The ATS model
itself is *not* in this file (see §3).

### Construction

```python
solver = ChebAr(Ham, jump_ops, T_block, jump_ops_LdL=None, output_phase=None,
                dims=None, cheb_degree=6, rtol=1e-9, atol=1e-10,
                require_gpu=True)
```

| Parameter | Meaning |
|---|---|
| `Ham`, `jump_ops` | `dynamiqs` Hamiltonian (possibly time-dependent) and Lindblad jump operators |
| `dims` | per-mode Hilbert dimensions, e.g. `(n_a, n_b)`; `N = prod(dims)`, Liouville dim `dim = N**2`. Inferred from `Ham.dims` if omitted |
| `jump_ops_LdL` | optional precomputed `L†L` operators; switches the propagator to `dq.mesolve_fast` (requires the pinned dynamiqs fork) |
| `output_phase` | optional per-basis-state phase applied after each block, to undo an interaction-frame rotation |
| `T_block` | duration of one propagation block (Floquet period); sets `tsave = [0, T_block]` and the rate normalization |
| `cheb_degree` | degree of the Chebyshev filtering polynomial |
| `rtol`, `atol` | `mesolve` integration tolerances |
| `require_gpu` | raises `RuntimeError` at construction if no JAX GPU device is visible (set `False` for a slow CPU fallback) |

### Method reference

| Method | Role |
|---|---|
| `make_x0(seed, hermitize=True)` | random complex starting vector, optionally Hermitized |
| `hermitize_vec(x0)` *(static)* | projects a vectorized operator onto its Hermitian part |
| `scipy_dominant_eig(x0=None, k=1, which="LR")` | reference dominant eigenpair via `scipy.sparse.linalg.eigs` (CPU, numpy) |
| `arnoldi_build_hessenberg(x0, polynomial, m_arnoldi)` | jitted plain Arnoldi factorization (modified Gram–Schmidt + reorthogonalization) |
| `first_estimation(x0, m_arnoldi=40)` | plain Arnoldi on `P`; returns `(Q, H, ritz_vals)` |
| `find_enclosing_ellipse(ritz_vals, vertex_x, margin, num_a)` *(static)* | smallest-area ellipse through `(vertex_x, 0)` enclosing the given Ritz values with a normalized clearance `margin` |
| `chebyshev_complex(n, z)` *(static)* | `T_n(z)` for complex `z` |
| `setup_chebyshev(ritz_vals, margin=4e-2, num_a=2000)` | finds the enclosing ellipse, builds the Chebyshev filter and the filtered/warm-startable Arnoldi; stores `self.ellipse` |
| `optimal_polynomial_ellipse(m, semi_re, semi_im, lambda_1, degree)` | scalar filtering polynomial normalized to `p(lambda_1)=1` for an arbitrary ellipse (utility, not used internally by `setup_chebyshev`'s main path) |
| `arnoldi_hessenberg(x0, polynomial, m_new, ..., m_old=0, Q_old=None, H_old=None, mu_list_old=None, warm_start=False)` | public, warm-restartable filtered Arnoldi; returns `(Q, H, mu_list)` and optionally checks `|mu_list[-1]| <= 1 + tol` |
| `recover_mu_list(H, m_new, m_old=0, mu_prev=None, warm_start=False)` | inverts the Chebyshev-filtered Ritz values back to eigenvalues of `P`, step by step, tracked by continuity |
| `ritz_vector(Q, H, m_arnoldi, which="abs", target=None)` | reconstructs the full Ritz vector/density matrix for a chosen (or continuity-tracked) eigenvalue |
| `residual_check(x_ritz)` | Rayleigh quotient `mu_RQ`, relative residual, and rate on the *true* (unfiltered) propagator |
| `rate_from_mu(mu)` | `-log(mu) / T_block` |

### Internal/private helpers

`_build_helpers` (jits `normalize`, `project_trace_zero_vec`), `_build_propagator`
(jits `propagate_block_projected`, the actual `mesolve`-based action of `P`),
`_build_chebyshev_filter` (builds `scaled_apply` and `chebyshev_filter`),
`_build_arnoldi_hessenberg` (builds the jitted warm-startable Arnoldi core),
`_invert_chebyshev`.

### Typical use (from the module docstring)

```python
from cheb_ar import ChebAr
from cheb_ar.models.ats import build_ats_hamiltonian

H, jump_ops, T_block, _ = build_ats_hamiltonian(alpha_sq=8.5)
solver = ChebAr(H, jump_ops, T_block, dims=(20, 11), cheb_degree=6)
x0 = solver.make_x0(seed=0)

Q0, H0, ritz_vals = solver.first_estimation(x0, m_arnoldi=40)
solver.setup_chebyshev(ritz_vals, margin=4e-2)

Q, H, mu_list = solver.arnoldi_hessenberg(x0, solver.chebyshev_filter, 40)
Q, H, mu_list = solver.arnoldi_hessenberg(
    x0, solver.chebyshev_filter, 80,
    m_old=40, Q_old=Q, H_old=H, mu_list_old=mu_list)
rate = solver.rate_from_mu(mu_list[-1])
```

> `build_ats_hamiltonian` and its rotating-/interaction-frame variants live in
> `src/cheb_ar/models/ats.py` (single source of truth, with the default
> experimental parameters as module constants; see `docs/models_ats.md`).

## 3. `scripts/` — sweep drivers

All three scripts share the same conventions. Every physical/numerical
constant is a CLI argument with the historical values as defaults; `--output`
is required; `--gpu-id` sets `CUDA_VISIBLE_DEVICES` before jax is imported
(which is why the heavy imports happen inside `main()`). The two Chebyshev
sweeps run in the **interaction frame**
(`build_ats_hamiltonian_interaction` + `mesolve_fast` via `jump_ops_LdL` +
`output_phase`; see `docs/models_ats.md`). Each sweep point is wrapped in
`try/except`: a failure is logged and appended as an `"error"` entry so one
bad point doesn't kill the sweep. Results are written as indented JSON via
`cheb_ar.io.to_jsonable` (see `docs/io.md`).

### `sweep_eps_p.py`

Sweeps the pump strength `eps_p` (default `linspace(0.1, 1.1, 6)`) at fixed
`alpha_sq`:

- `kappa_b` is *not* fixed across the sweep — it is rescaled at each point as
  `kappa_b = sin(eps_p)/sin(eps_p_init) * kappa_b_init` to keep the adiabatic
  ratio `kappa_b / g` roughly constant.
- The first point — and any point right after a failure — starts cold with
  `--m-arnoldi-first` Krylov vectors; every other point uses the smaller
  `--m-arnoldi` and is warm-started from the previous point's `x_ritz`
  (`warm_start=True`, i.e. continuity-tracked mu recovery). The warm vector is
  re-expressed in the new point's eigenbasis via `transform_vectorized_state`,
  since the interaction-frame basis depends on `eps_p`. `--no-warm-start`
  cold-starts every point.

### `sweep_alpha.py`

Sweeps the cat size `alpha_sq` at fixed `eps_p` and fixed `kappa_b` (default:
the model's `KAPPA_B`), same pipeline. Warm starting is optional and comes
from a *previous results file* (`--warm-start-file`; its i-th `x_ritz` seeds
the i-th point, with `warm_start=True`). The vectors are used as-is — the
interaction-frame eigenbasis does not depend on `alpha_sq` — so the file must
come from a run with the same `n_a`/`n_b`, ideally the same `eps_p`.

### The ellipse-fit escalation ladder

Both sweeps guard `first_estimation` + `setup_chebyshev` with the same
escalation ladder (identical inline code in both scripts, per-script on
purpose — no shared driver):

1. Try `m_arnoldi_0`, then `round(m_arnoldi_0 * sqrt(2))`, then
   `2 * m_arnoldi_0` Krylov vectors, each with the requested `--margin`.
2. If the fit still fails, keep the last (largest) estimation and halve the
   margin repeatedly, down to a floor of `1e-5`.
3. If even the first estimation itself never succeeded, the point fails.

The `m_arnoldi_0` and `margin` recorded in each result entry are the values
**actually used**, not the CLI values.

### `exact_diagonalization.py`

Brute-force reference for small cats: sweeps `eps_p` × `alpha_sq` on a small
Hilbert space (default `n_a=13`, `n_b=6`), builds the full one-period
propagator with `dq.mepropagator` on the **rotating-frame** Lindbladian
(`build_ats_hamiltonian_rotating`), diagonalizes it exactly, takes the
second-largest-`|mu|` eigenvalue as the bit-flip eigenvalue, and converts via
`-log(mu) / T_block`.

## 4. Dependencies

- `jax` / `jax.numpy` (with `jax_enable_x64` — the whole pipeline is
  double precision), GPU required for practical runtimes.
- [`dynamiqs`](https://github.com/dynamiqs/dynamiqs) for operators, states,
  and the `mesolve` time-dependent Lindblad master-equation integrator
  (`dq.method.Tsit5` here).
- `numpy`, `scipy` (`scipy.sparse.linalg.{LinearOperator,eigs}`,
  `scipy.special.jv`).
- `json` (sweep-script output only).

## 5. Layout

The reorganization sketched here during the merge is complete: the
model-agnostic solver lives in `src/cheb_ar/solvers/cheb_ar.py`, the ATS
builders and constants in `src/cheb_ar/models/ats.py` (`docs/models_ats.md`),
the JSON helpers in `src/cheb_ar/io.py` (`docs/io.md`), and the CLI sweep
drivers in `scripts/` with their OAR job files in `scripts/cluster/`. The GPU
test notebooks — validated on the cluster — live in `tests/`; the pre-merge
code is kept untracked in `Archive/`. The sweep scripts deliberately stay
separate (no unified driver); when they diverge, `sweep_alpha.py` is aligned
on `sweep_eps_p.py`.
