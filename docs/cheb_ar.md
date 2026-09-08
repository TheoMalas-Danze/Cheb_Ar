# Chebyshev–Arnoldi bit-flip rate solver

Documentation for `src/cheb_ar/solvers/cheb_ar.py` (the `ChebAr` solver) and
`Cheb_Ar_old/loop_eps_p/cheb_ar_loop_eps_p.py` (a sweep script built on top of
its older two-mode version, now kept in `Archive/cheb_ar.py`, untracked). Both
originate from `full_ATS/Chebyshev-Arnoldi.ipynb` and are GPU-only (JAX +
`dynamiqs`); they cannot be executed on a CPU-only machine.

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
# model builders live outside the solver (future cheb_ar.models)

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

> **Resolved:** `build_ats_hamiltonian` and its rotating-/interaction-frame
> variants now live in `src/cheb_ar/models/ats.py` (single source of truth,
> with the default experimental parameters as module constants). The copies in
> the `Cheb_Ar_old` sweep scripts remain only until those scripts are replaced
> by the unified CLI driver.

## 3. `cheb_ar_loop_eps_p.py` — sweep script

Loads `ChebAr` and runs the full pipeline over a sweep of the pump strength
`eps_p`, warm-starting each point from the Ritz vector of the previous one.

### `build_ats_hamiltonian(...)`

Builds the driven full-ATS Lindbladian in the rotating frame (mirrors the
Hamiltonian cells of the original notebook):

- Two bosonic modes `a` (storage, dim `n_a`) and `b` (buffer, dim `n_b`).
- Couplings `g`, `g2` derived from the pump amplitude `epsilon_p` and Josephson
  parameters `E_J`, `phi_a`, `phi_b`.
- Two-photon dissipation rate `kappa_2 = 4*g**2/kappa_b`, plus a single-photon
  loss `kappa_1 = 0.005 * kappa_2`.
- A rotating-frame drive `H_drive` at `w_b = 2*w_a` with amplitude
  `epsilon_d = 2*alpha_sq*g2`, plus the non-linear Josephson term built from
  `dq.sinm(phi_a_tot + phi_b_tot)`.
- Returns `(Ham, jump_ops, T_block, params)` with `T_block` = one drive period
  (`n_periods * 2*pi / w_a`) and `params` holding the derived quantities for
  bookkeeping/output.

### `run_for_epsp(eps_p, kappa_b, m_arnoldi, x0=None)`

Runs one sweep point end to end: build the Hamiltonian → `ChebAr(...)` →
`first_estimation` (fixed `m_arnoldi_0`) → `setup_chebyshev` → filtered
`arnoldi_hessenberg` (`warm_start=True` if an `x0` was supplied) →
`rate_from_mu` → `ritz_vector` → `residual_check`. Returns two dicts:
`result_raw` (JAX/complex objects, for chaining warm starts in-process) and
`result_json` (fully JSON-serializable, via `_to_jsonable`, for saving to
disk).

### `_to_jsonable(obj)`

Recursive converter: JAX/NumPy arrays → nested lists, complex numbers →
`{"real": ..., "imag": ...}`, NumPy scalars → plain Python `int`/`float`.

### `__main__` sweep driver

- Sweeps `eps_p` over `np.linspace(0.1, 1.1, 6)`.
- `kappa_b` is *not* fixed across the sweep — it is rescaled at each point as
  `kappa_b = sin(eps_p)/sin(eps_p_init) * kappa_b_init` to keep the adiabatic
  ratio `kappa_b / g` roughly constant.
- The first point uses `m_arnoldi_first` Krylov vectors and a fresh start
  (`x0=None`); every subsequent point uses the smaller `m_arnoldi_generic` and
  is warm-started from the previous point's `x_ritz` (`results_raw[idx_eps]`).
- Failures are caught per-point (`try/except`), logged with the exception
  type/message, and appended as an `"error"` entry so one bad point doesn't
  kill the whole sweep — except the *first* point, where a failure currently
  leaves `result` referencing the exception branch's undefined variable (see
  below).
- Results are written as indented JSON to `OUTPUT_PATH`.

### Hardcoded values to parameterize when cleaning up

These are fine for a one-off cluster run but should become CLI arguments,
a config file, or at least named constants at the top of a `main()`, so the
script is reusable without editing source:

- `sys.path.insert(0, "/home/tmalasda/dev")` and, in `cheb_ar.py`,
  `sys.path.insert(0, "/home/tmalasda/dynamiqs")` plus the matching
  `assert dq.__file__ == ...` — both are absolute paths tied to one user's
  cluster account and will break on any other machine/environment. Replace
  with a normal installed package (`pip install -e .` / proper `dynamiqs`
  dependency) rather than `sys.path` surgery.
- `OUTPUT_PATH = "/home/tmalasda/output/Cheb_Ar/21_07_2026/..."` — absolute,
  user- and date-specific output path.
- Physical/numerical constants defined at module scope: `n_a`, `n_b`,
  `cheb_degree`, `m_arnoldi_0`, `m_arnoldi_first`, `m_arnoldi_generic`,
  `margin`, `alpha_sq`, `kappa_b_init`, `eps_p_init`, `E_J`, `phi_a`, `phi_b`,
  `g_init`, `adiabatic_constant`. Several of these (e.g. `w_a`, `kappa_b`,
  `E_J`, `phi_a`, `phi_b`) are *also* default arguments of
  `build_ats_hamiltonian`, so the physical model parameters currently live in
  two places that can silently drift out of sync.
- `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` is present but commented out —
  worth turning into a real CLI flag (`--gpu-id`) instead of a comment to
  edit by hand.

### Known correctness note

If `run_for_epsp` raises on the *first* sweep point, the `except` block
appends an error dict but the code right after it (`print(f"rate_bf =
{result['rate_bf']}")`, `results.append(result)`) still runs unconditionally
and will raise a `NameError`/`KeyError` because `result` was never assigned.
Worth wrapping that reporting block in the same `try` or guarding it with a
flag when refactoring.

## 4. Dependencies

- `jax` / `jax.numpy` (with `jax_enable_x64` — the whole pipeline is
  double precision), GPU required for practical runtimes.
- [`dynamiqs`](https://github.com/dynamiqs/dynamiqs) for operators, states,
  and the `mesolve` time-dependent Lindblad master-equation integrator
  (`dq.method.Tsit5` here).
- `numpy`, `scipy` (`scipy.sparse.linalg.{LinearOperator,eigs}`,
  `scipy.special.jv`).
- `json` (sweep-script output only).

## 5. Suggested placement in a reorganized repo

- `src/cheb_ar/solvers/cheb_ar.py` — the model-agnostic `ChebAr` class.
  **Done** (installable via `pip install -e .`; the older two-mode version is
  archived, untracked, in `Archive/cheb_ar.py`).
- `src/cheb_ar/models/ats.py` — `build_ats_hamiltonian` and its
  rotating-/interaction-frame variants (moved out of the sweep scripts and
  notebooks; resolves the note in §2). **Done** (plus `cheb_ar.io` for the
  JSON helpers).
- `scripts/sweep_eps_p.py` — the `__main__` sweep driver, rewritten around
  `build_ats_hamiltonian` + `ChebAr` from the two modules above, with the
  hardcoded paths/constants in §3 promoted to CLI arguments or a config file.
  **To do.**
