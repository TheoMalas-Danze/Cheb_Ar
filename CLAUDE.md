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

## Repo layout (target structure post-merge proposal)

```
src/
  solvers/      # model-agnostic numerics, e.g. cheb_ar.py (ChebAr class)
  models/       # physical model builders, e.g. build_ats_hamiltonian
scripts/        # sweep/entry-point scripts (CLI-driven, no hardcoded paths)
docs/           # reference docs per module/subsystem
notebooks/      # exploratory notebooks (not imported by src/ or scripts/)
```

If a file doesn't yet live where this layout says it should, that's expected
mid-merge — move things opportunistically rather than all at once.

## Known issues to watch for while cleaning up

- **Hardcoded absolute paths**, e.g. `sys.path.insert(0, "/home/tmalasda/...")`
  and cluster-specific output paths. Replace with a proper installed
  dependency / relative path / CLI argument — flag any new ones you find.
- **Duplicated physics constants**: some parameters (e.g. `w_a`, `kappa_b`,
  `E_J`, `phi_a`, `phi_b`) are set as both module-level constants in sweep
  scripts *and* as default arguments of model-builder functions like
  `build_ats_hamiltonian`. These should have a single source of truth —
  don't add a third copy.
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

- `docs/cheb_ar.md` — full reference for the `ChebAr` solver and the
  `eps_p` sweep script (algorithm, method table, hardcoded values to fix).
- `src/cheb_ar/models/ats.py` — the three frame variants of the ATS builder
  and the default experimental constants (single source of truth).
- (Add one entry here per module/doc as the merge and cleanup progress.)
