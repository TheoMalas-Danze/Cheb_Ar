# CLAUDE.md

Guidance for Claude Code when working in this repository. Keep this file
short — deep dives belong in `docs/`.

## What this repo is

Simulation code for a driven, dissipative full-ATS (Asymmetrically Threaded
SQUID) system: two coupled bosonic modes, Floquet driving, Lindblad
dissipation. The main physics question is the **bit-flip rate** of the
system, extracted as the leading eigenvalue of the one-period propagator of
the master equation via a custom Chebyshev-accelerated Arnoldi solver
(`ChebAr`, see `docs/cheb_ar.md`).

This repo is a merge of an older codebase and a newer one; some duplication,
inconsistent conventions, and dead code are expected until cleanup is done.

## Hard constraints

- **The code requires a GPU.** It's built on `jax` (double precision,
  `jax_enable_x64`) and `dynamiqs`'s `mesolve`, and several entry points
  explicitly `raise RuntimeError` if no JAX GPU device is visible
  (`require_gpu=True` is the default in `ChebAr`). **Do not try to run,
  execute, or "verify" this code locally** — there is no CPU fallback that
  finishes in reasonable time. Static review (reading, refactoring, tracing
  imports) is the only mode available outside the cluster.
- Because nothing can be executed here, **prefer small, isolated, easily
  diffable commits** over large rewrites, so correctness can be checked by
  inspection and, eventually, by the user re-running on the cluster.
- Do not remove or "simplify" numerical tolerances, filtering margins, or the
  physical constants in the Hamiltonian builders without flagging it — these
  encode real experimental parameters (Josephson energies, phases, drive
  strengths), not arbitrary defaults.

## Repo layout

```
src/cheb_ar/
  solvers/      # model-agnostic numerics: cheb_ar.py (ChebAr class)
  models/       # physical model builders: ats.py (3 frame variants + constants)
  io.py         # JSON (de)serialization helpers for sweep results
scripts/        # sweep/entry-point scripts (CLI-driven, no hardcoded paths)
  cluster/      # OAR job files
docs/           # reference docs per module/subsystem
tests/          # GPU test notebooks, validated on the cluster (not imported by src/ or scripts/)
Archive/        # untracked: pre-merge code kept locally, just in case
```

Installable package: `pip install -e .` (that's what requirements.txt does).

## Known issues to watch for while cleaning up

- **Hardcoded absolute paths** (e.g. `sys.path.insert(0, "/home/...")`,
  cluster output paths) were removed in the merge cleanup — flag any new
  ones you find; use CLI arguments instead.
- **Physics constants** have their single source of truth in
  `src/cheb_ar/models/ats.py` (module constants + builder defaults) —
  don't introduce copies in scripts or notebooks.
- **Docstrings can lag the actual module layout** 
- GPU-guard `RuntimeError`s (`require_gpu=True`, explicit `jax.devices()`
  checks) are intentional safety checks, not leftover debug code — keep them
  unless the user asks to relax them.

## Conventions

- Double precision throughout (`jax_enable_x64`); don't introduce
  `float32`/`complex64` paths without an explicit reason.
- Solvers are kept model-agnostic (operate on `(Ham, jump_ops)` +
  dimensions); model-specific physics stays in `models/`, not in solver code.
- JIT boundaries matter for performance on GPU — don't casually wrap/unwrap
  `@jax.jit` functions during refactors without checking what's static vs
  traced (`static_argnames` usage in this codebase is deliberate).

## Where to look for more detail

- `docs/cheb_ar.md` — full reference for the `ChebAr` solver and the sweep
  scripts (algorithm, method table, ellipse-fit escalation ladder).
- `docs/models_ats.md` — the three frame variants of the ATS builder in
  `src/cheb_ar/models/ats.py` and the default experimental constants
  (single source of truth).
- `docs/io.md` — the JSON (de)serialization helpers in `src/cheb_ar/io.py`
  and their round-trip caveats.
