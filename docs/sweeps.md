# Sweeps: scripts and notebooks

How the bit-flip rate is swept over the cat size `alpha_sq` (= `|alpha|^2`,
see `docs/models_ats.md` §2) and the pump strength `eps_p`. Every sweep point
goes through one of the per-solver pipelines in `src/floquet_lindblad/pipeline/`:

| pipeline | solver | role |
|---|---|---|
| `solve_point_ks` / `solve_point_ks_safe` (`pipeline/krylov_schur.py`) | `KrylovSchurLindblad` (`docs/krylov_schur.md`) | **main** |
| `solve_point_cheb_ar` / `solve_point_cheb_ar_safe` (`pipeline/cheb_ar.py`) | `ChebAr` (`docs/cheb_ar.md`) | secondary, cross-check |

Both share one calling contract — sweep coordinates, an `x0` / `V_prev` warm
start (the vector is rotated into the new eigenbasis with
`transform_vectorized_state` when `V_prev` is given), `want_x_ritz` / `want_V`
to ship the chain state back — and return plain numpy / python, so a result
crosses to a client without the GPU stack. `rate_bf` is the rate in both; the
final residual is `res_rel_n` (Krylov-Schur) or `res["res_rel"]` (ChebAr). A
sweep submits the `*_safe` form: a numerical failure comes back as an
`"error"` entry carrying the point's coordinates, instead of raising and
sinking the sweep.

## 1. Scripts (`scripts/`)

Cluster drivers: they run *on* the anb-compute Ray cluster, one GPU task per
point, and are submitted from anb-dev with `scripts/cluster/submit.py <name>
-- <args>`. Heavy imports happen inside `main()`; `--output` (a `gs://`
prefix) is required, since the job's disk dies with the job. Results are
written as indented JSON via `floquet_lindblad.io.to_jsonable` (`docs/io.md`),
each entry tagged with the `solver` that wrote it.

`sweep_alpha.py` and `sweep_eps_p.py` take `--solver {krylov_schur,cheb_ar}`
(default `krylov_schur`) plus one argument group per solver; `--n-a` / `--n-b`
default per solver. They stay separate on purpose (no unified driver); when
they diverge, `sweep_alpha.py` is aligned on `sweep_eps_p.py`.

### `sweep_alpha.py`

Sweeps `alpha_sq` at fixed `eps_p` and fixed `kappa_b` (default: the model's
`KAPPA_B`). The points are independent — the interaction-frame eigenbasis does
not depend on `alpha_sq` — so they fan out, one task each.

- Krylov-Schur defaults: band 3 of the ladder in
  `notebooks/sweeps/sweep_alpha.ipynb` (`(n_a, n_b) = (35, 16)`, `m / k = 50 /
  12`, loop `1e-9 / 1e-10`, final `1e-11 / 1e-12`, `res_tol = 1e-10`, 50-period
  final block), which covers `alpha_sq < 6.5`. The script uses one setting for
  every point; the notebook's ladder adapts it per point.
- ChebAr defaults: unchanged from before (`(20, 8)`, `cheb_degree = 6`,
  `m_arnoldi_0 / m_arnoldi = 60 / 80`, `margin = 5e-3`).
- Optional warm start from a *previous results file* (`--warm-start-file`; its
  i-th `x_ritz` seeds the i-th point). The vectors are used as-is, so the file
  must come from a run with the same `n_a` / `n_b`, ideally the same `eps_p`.

### `sweep_eps_p.py`

Sweeps `eps_p` (default `linspace(0.1, 1.1, 6)`) at fixed `alpha_sq`:

- `kappa_b` is rescaled at each point as `kappa_b = sin(eps_p)/sin(eps_p_init)
  * kappa_b_init`, which keeps `kappa_b / g` — and, since `epsilon_d ∝ g`, also
  `epsilon_d / kappa_b` — exactly constant.
- The first point, and any point right after a failure, starts cold with the
  larger Krylov size (`--m`, Krylov-Schur; `--m-arnoldi-first`, ChebAr); every
  other point is warm-started from the previous point's `x_ritz`, rotated into
  its eigenbasis, with the smaller size (`--m-warm`; `--m-arnoldi`).
  `--no-warm-start` cold-starts every point, and they then run in parallel.
- Krylov-Schur defaults: those of `notebooks/sweeps/sweep_eps_p.ipynb` (`(30,
  15)`, sized for the first — hardest — point and held along the sweep), with
  the final block shortened as the gap opens (`n_blocks_auto`, down from the
  `--n-blocks-final` cap to what gives `1 - mu**n ~ --rq-resolution`).
- `--kappa-b-init` defaults to `0.6 / 10.4`, **not** `ats.KAPPA_B = 5 / 10.4`
  (historical; the notebooks use `ats.KAPPA_B`).

### ChebAr only: the ellipse-fit escalation ladder

`solve_point_cheb_ar` guards `first_estimation` + `setup_chebyshev` with an
escalation ladder (`pipeline.cheb_ar.escalating_setup`):

1. Try `m_arnoldi_0`, then `round(m_arnoldi_0 * sqrt(2))`, then
   `2 * m_arnoldi_0` Krylov vectors, each with the requested `--margin`.
2. If the fit still fails, keep the last (largest) estimation and halve the
   margin repeatedly, down to a floor of `1e-5`.
3. If even the first estimation itself never succeeded, the point fails.

The `m_arnoldi_0` and `margin` recorded in each result entry are the values
**actually used**, not the CLI values.

### `exact_diagonalization.py`

Brute-force reference for small cats, independent of both pipelines: sweeps
`eps_p` × `alpha_sq` on a small Hilbert space (default `n_a=13`, `n_b=6`),
builds the full one-period propagator with `dq.mepropagator` on the
**rotating-frame** Lindbladian (`build_ats_hamiltonian_rotating`),
diagonalizes it exactly, takes the second-largest-`|mu|` eigenvalue as the
bit-flip eigenvalue, and converts via `-log(mu) / T_block`.

## 2. Notebooks (`notebooks/sweeps/`)

The same sweeps, run from a notebook with `acr.run` and plotted locally.

| notebook | solver | shape |
|---|---|---|
| `sweep_alpha.ipynb` | Krylov-Schur | fan-out; truncation and tolerances follow a ladder in `alpha_sq`; perturbative comparison |
| `sweep_eps_p.ipynb` | Krylov-Schur | warm-started chain; adiabatic-ratio and warm-start checks |
| `sweep_alpha_eps_p.ipynb` | Krylov-Schur | `alpha_sq` chains fanned out, each a warm `eps_p` chain; run twice at two truncations as a convergence check |
| `legacy_cheb_ar/sweep_alpha.ipynb`, `legacy_cheb_ar/sweep_eps_p.ipynb` | ChebAr | the pre-Krylov-Schur forms of the first two |

The notebook client cannot import `floquet_lindblad` (it pulls in jax and
dynamiqs at module scope), so each notebook hands the coordinator a thin
`solve_point_ks_safe` whose body imports the package version on the worker.
The per-point *settings* (the ladders) stay in the notebooks: they are
per-study calibrations, not part of the pipeline.
