# Migrating Cheb_Ar from OAR/`chuc` to the anb-compute Ray cluster

> **Cat-size convention.** The runs below predate the 2026-09 fix of
> `epsilon_d` (see `docs/models_ats.md` §2): here `alpha_sq` is twice the
> cat size, `|alpha|^2 ~ alpha_sq / 2`. The same physical points are now
> `alpha_sq` = half the values quoted.

Everything below marked *measured* was verified by running against the live
cluster on 2026-09-17 (`scripts/cluster/smoke_gpu.py`, `scripts/cluster/probe_dq.py`,
`scripts/cluster/probe_fork.py`, `scripts/cluster/smoke_sweeps.py`).

Reference docs: `alice-bob/theory/hardware/anb-emulator`, under
`src/anb-compute/docs/user_guide/` (`ray.md`, `ray_setup.md`, `ray_api.md`,
`ray_packaging.md`, `ray_scheduling.md`).

## Status

| § | Item | State |
| --- | --- | --- |
| 2 | `mesolve_fast` rebased onto v0.3.6, pin updated | **done** |
| 3 | `--gpu-id` / `CUDA_VISIBLE_DEVICES` removed | **done** |
| 4 | `notebooks/solvers/cheb_ar.ipynb` → one `acr.run` | **done, runs green** |
| 5 | Sweeps → one job + fan-out | **done** |
| 6 | Results → GCS | **done** |
| 8 | The other three test notebooks | **done** |

## New layout

The sweep pipeline was duplicated four times, in near-identical form, across
the two sweep scripts and the two sweep notebooks. It now lives once in
[`src/floquet_lindblad/pipeline.py`](../src/floquet_lindblad/pipeline.py), which all four import —
and which cluster workers import too, since it travels in `py_modules`.

| | What it is |
| --- | --- |
| `floquet_lindblad.pipeline.solve_point` | One sweep point end to end. Returns **plain numpy/python** so a result can cross to a client with no jax/dynamiqs. |
| `floquet_lindblad.pipeline.solve_point_safe` | The same, returning a failure instead of raising it. This is what a sweep submits as its task. |
| `floquet_lindblad.pipeline.escalating_setup` | The m-schedule / margin-halving ladder, extracted verbatim. |
| `scripts/sweep_*.py`, `scripts/exact_diagonalization.py` | **Cluster drivers.** They run *on* the cluster, fan out, and save to GCS. |
| `scripts/cluster/submit.py` | The thin client that submits a driver. Replaces the OAR `.sh` files. |

```bash
pixi run python scripts/cluster/submit.py sweep_alpha -- --n-a 20 --n-b 8
```

It invents a fresh dated GCS prefix per run (`save_output` overwrites silently,
so prefixes are never reused), prints the job id, and prints the `load_output`
snippet to read the results back.

## Headline

**The environment is almost entirely a non-problem.** The GPU worker image
already carries jax 0.10.2 with working CUDA, diffrax 0.7.2, dynamiqs 0.3.6,
qutip and matplotlib — i.e. every pin in `pyproject.toml` except one.

The real work is structural, not environmental:

1. One missing function: `dq.mesolve_fast` (§2).
2. `CUDA_VISIBLE_DEVICES` handling is actively harmful here (§3).
3. Notebook/script control flow has to be reshaped for a submit-and-wait model (§4, §5).
4. Results must go to GCS, not a local path (§6).

## 1. What the GPU worker actually has (measured)

A GPU task runs in `/anb-envs/.pixi/envs/**prod-gpu**/`, which is a *different*
environment from the `prod` one the driver runs in. This is not mentioned in the
user guide, and it matters: the `prod` environment's dependency list
(`anb-core-dependencies[ml]`) suggests a bare CPU jax and no dynamiqs, which is
misleading. Check `prod-gpu`.

| Cheb_Ar pin | On `prod-gpu` | Verdict |
| --- | --- | --- |
| `jax==0.10.2` | 0.10.2 | exact match |
| `jaxlib==0.10.2` | 0.10.2 | exact match |
| `diffrax==0.7.2` | 0.7.2 | exact match |
| `dynamiqs` (fork) | 0.3.6 (upstream) | **missing `mesolve_fast`** — see §2 |
| `numpy`, `scipy` | scipy 1.17.1 | fine |
| `matplotlib` | 3.11.0 | present |
| (dynamiqs dep) `qutip` | 5.2.3 | present |
| `floquet_lindblad` (then `cheb_ar`) | — | ship it (§4) |

GPU, measured:

```
default_backend: gpu          devices: ['gpu:0']
jax-cuda12-plugin, jax-cuda12-pjrt, nvidia-cuda-{cccl,cupti,nvcc,nvrtc,runtime}-cu12
x64_dtype: float64            matmul_device: cuda:0   (complex128, correct result)
driver: 580.159.04, x86_64
```

So double-precision complex linear algebra runs on the GPU out of the box.
`jax_enable_x64` survives, which is the one thing that would have quietly
wrecked the numerics. **`ChebAr(require_gpu=True)` will pass on this image** —
no `jax[cuda12]` in `runtime_env["pip"]`, no image rebuild.

!!! note "The user guide's GPU snippet is broken"
    `ray.md`'s "Did I really get a GPU?" recipe shells out to `nvidia-smi -L`,
    but that binary is **not on PATH** in the worker image: it raises
    `FileNotFoundError` and tells you nothing. Use `ray.get_gpu_ids()`,
    `/proc/driver/nvidia/version` and `jax.devices()` instead, as
    `scripts/cluster/smoke_gpu.py` does. Worth reporting upstream.

## 2. The one real dependency gap: `mesolve_fast`

Measured on the image: `hasattr(dq, "mesolve_fast") is False`.

[`ChebAr._build_propagator`](../src/floquet_lindblad/solvers/cheb_ar.py#L202) calls
`dq.mesolve_fast` whenever `jump_ops_LdL` is passed — which is the entire
interaction-frame path, i.e. what `notebooks/solvers/cheb_ar.ipynb` and both sweeps use. So
this blocks the port.

Where it comes from: `pyproject.toml` pins
`dynamiqs @ git+.../TheoMalas-Danze/dynamiqs.git@b52cb66`. That fork is **one
commit** ("Adding LdL as an optional parameter for mesolve", 8 files, +81/−10)
on top of upstream `108fcd5`.

The awkward part: `108fcd5` sits between upstream `v0.3.5` and `v0.3.6`, so the
image's 0.3.6 is **10 commits ahead of the fork's base**. Shipping the fork
wholesale therefore *downgrades* dynamiqs on the worker, and two of those ten
commits are API changes:

```
a49b30f Bump version to 0.3.6
956d069 Api/expose progress meter (#1090)
cf17472 Docs : flatten solver options into keyword arguments (#1089)
e56b242 API: flatten solver options into keyword arguments (#1088)
cd76200 fix: use floor division in integrate_by_chunks (#1093)
4a15b5c Fix ClipStepSizeController replace() crash with discontinuous Hamiltonians (#1086)
243a7de Add typing overloads for destroy, create, and number (#1075)
faec9c8 Optimize PWC prefactor evaluation (#1084)
f26441b Fix CI dependency resolution for taskipy and ty typing issues (#1087)
c3bf215 Fix type checking with latest version (#1076)
```

Do **not** put the git URL in `runtime_env["pip"]`: it resolves and installs on
every node at every job start, it drags in qutip (compiled) which is already on
the image, and it assumes worker egress to github, which is unverified.

### DONE: the commit has been rebased onto v0.3.6

Checkout at `../dynamiqs` (beside this repo; override with `$DYNAMIQS_SRC`), branch **`precomputed-ldl-v036`**, commit
`af64c63d`, based directly on `v0.3.6`. The cherry-pick itself applied with **no
conflicts** and was byte-identical to the original; the follow-up API adaptation
below is folded into the same commit. The original, unrebased patch is still
preserved on `origin/tmalasdanze/precomputed-ldl` (`b52cb66`), untouched.

Pushed to
[`TheoMalas-Danze/dynamiqs@precomputed-ldl-v036`](https://github.com/TheoMalas-Danze/dynamiqs/tree/precomputed-ldl-v036),
full sha `af64c63df011d09633f158d06d5f5b3ee637cf9e`. So the `pyproject.toml` pin
becomes:

```toml
"dynamiqs @ git+https://github.com/TheoMalas-Danze/dynamiqs.git@af64c63df011d09633f158d06d5f5b3ee637cf9e",
```

That pin is for the *local* env (and for anyone installing the repo normally).
For cluster jobs the checkout travels via `py_modules` by path instead — the
image already supplies dynamiqs' own dependencies.

Verified on a real GPU worker by shipping it through `py_modules`
(`scripts/cluster/probe_fork.py`):

```
dynamiqs_file: .../runtime_resources/py_modules_files/.../dynamiqs/__init__.py
has_mesolve_fast: True     backend: gpu     state_dtype: complex128
max_abs_diff_vs_mesolve: 0.0     agrees: True
n_final: 0.8105784   n_expected (exp(-kappa t)): 0.8105842
```

So the shipped copy does shadow the image's 0.3.6, `mesolve_fast` exists, runs in
complex128 on the GPU, and agrees exactly with plain `mesolve`.

### ...but v0.3.6 *did* break one call style, in the package itself

Upstream #1088 flattened the **public** `mesolve`'s solver options into keyword
arguments. On v0.3.6:

```python
dq.mesolve(H, Ls, rho0, tsave, method=..., options=dq.Options(...))   # TypeError
dq.mesolve(H, Ls, rho0, tsave, method=..., assume_hermitian=False)    # correct
```

`mesolve_fast` was technically *not* forced to change — it calls the internals
(`_mesolve`) directly, and those still take an `Options`. But leaving it with
`options=` while its sibling `mesolve` used flattened kwargs made the fork
internally inconsistent, so **`mesolve_fast` has been flattened to match**
(folded into commit `af64c63d`):

```python
dq.mesolve_fast(H, jump_ops, jump_ops_LdL, rho0, tsave, *,
                method=..., gradient=None, save_states=True,
                progress_meter=None, t0=None, save_extra=None,
                assume_hermitian=True)
```

`cartesian_batching` and `vectorized` are deliberately **not** exposed:
`mesolve_fast` calls the integrator directly and does not vectorize, so accepting
them would silently do nothing. Passing them now raises `TypeError` (verified),
which is the honest failure.

So **all three** call sites in
[`cheb_ar.py`](../src/floquet_lindblad/solvers/cheb_ar.py) change the same way:

| Site | Call | Change |
| --- | --- | --- |
| [L203–210](../src/floquet_lindblad/solvers/cheb_ar.py#L203) | `dq.mesolve_fast(..., options=dq.Options(assume_hermitian=False))` | → `assume_hermitian=False` |
| [L213–219](../src/floquet_lindblad/solvers/cheb_ar.py#L213) | `dq.mesolve(..., options=dq.Options(assume_hermitian=False))` | → `assume_hermitian=False` |
| [L258–263](../src/floquet_lindblad/solvers/cheb_ar.py#L258) | `dq.mesolve(..., options=dq.Options(assume_hermitian=False))` (in `scipy_dominant_eig`) | → `assume_hermitian=False` |

i.e. drop `options=dq.Options(...)` and pass `assume_hermitian=False` directly.

`dq.mepropagator(Ham_full, jump_ops, tsave)` in
[`exact_diagonalization.py:63`](../scripts/exact_diagonalization.py#L63) passes no
options and its positional signature is unchanged — unaffected.

The `opts = dq.Options(assume_hermitian=False)` line in the two test notebooks is
dead (never passed anywhere); it can just go.

This is worth flagging loudly because the plain-`mesolve` branch is the *fallback*
path, so a `TypeError` there would only surface the first time someone constructs
a `ChebAr` without `jump_ops_LdL` — likely much later, and confusingly.

### Remaining options for the longer term

1. **Upstream the commit**, then drop the git pin for `dynamiqs>=<release>` and
   ship nothing at all. Cleanest end state — one small, generally useful patch.
   Note this also needs the anb-compute image to pick up that release.
   If you do, consider flattening `mesolve_fast`'s own `options=` to match
   v0.3.6's public style; maintainers will likely ask for it.
2. Keep shipping `precomputed-ldl-v036` via `py_modules` by path. Ray
   content-hashes the directory, so edits reship automatically. This is the
   working state today.
3. Fallback needing nothing shipped but `floquet_lindblad`: pass `jump_ops_LdL=None` so
   `_build_propagator` takes the plain `dq.mesolve` branch (after fixing the
   `options=` call above). Slower, and not what the sweeps are tuned for, but
   fine for a first end-to-end smoke test.

## 3. `CUDA_VISIBLE_DEVICES` must go

Four places set it:

- [`notebooks/solvers/cheb_ar.ipynb`](../notebooks/solvers/cheb_ar.ipynb) cell 1:
  `os.environ["CUDA_VISIBLE_DEVICES"] = "2"`
- [`scripts/sweep_alpha.py:48`](../scripts/sweep_alpha.py#L48),
  [`scripts/sweep_eps_p.py:48`](../scripts/sweep_eps_p.py#L48),
  [`scripts/exact_diagonalization.py:30`](../scripts/exact_diagonalization.py#L30):
  a `--gpu-id` flag applied before the heavy imports.

These are `chuc`-era artifacts for picking a card on a shared box. **Ray sets
`CUDA_VISIBLE_DEVICES` itself** to the GPU it allocated. Measured on a real task:

```
ray_gpu_ids: [3]        CUDA_VISIBLE_DEVICES: 3
/dev/nvidia0 ... /dev/nvidia15          # 16 GPUs on the node, not 4
```

So the node is far denser than the docs' "4 GPU" description, the index you get
is arbitrary, and hardcoding `"2"` would point at a card allocated to somebody
else's task. Drop `--gpu-id` and the `os.environ` line entirely.

Keep the `require_gpu=True` guard in
[`ChebAr.__init__`](../src/floquet_lindblad/solvers/cheb_ar.py#L101). It is exactly right
here — it turns "silently ran on CPU for six hours" into an immediate
`RuntimeError`.

## 4. The notebook: 13 stateful cells → one function

[`notebooks/solvers/cheb_ar.ipynb`](../notebooks/solvers/cheb_ar.ipynb) is a linear pipeline
spread across cells, each mutating `solver` for the next:

```
build_ats_hamiltonian_interaction()  ->  ChebAr(...)  ->  first_estimation
  ->  setup_chebyshev  ->  arnoldi_hessenberg  ->  rate_from_mu / ritz_vector
  ->  residual_check   ->  rho_lab = V @ x_ritz.reshape(N,N) @ V.conj().T
  ->  dq.plot.wigner(...)
```

`acr.run` ships **one** picklable function and returns **its** value. Cell-to-cell
state does not survive, and `solver` cannot live on your side between calls: it
holds jitted jax closures and dynamiqs operators, none of which are
cloudpickle-clean.

Also drop cell 0, `%pip install -r ../requirements.txt`: `pip` is `pixi` on
anb-dev, and a local install does nothing for a worker anyway.

So: one function doing build → estimate → filter → Arnoldi → rate → residual,
returning **plain arrays**; then reconstruct `rho_lab` and plot locally.

```python
def run_point(alpha_sq, n_a=25, n_b=11, m_arnoldi_0=60, m_arnoldi=120, margin=1e-2):
    from floquet_lindblad import ChebAr
    from floquet_lindblad.io import to_jsonable
    from floquet_lindblad.models.ats import build_ats_hamiltonian_interaction

    H_I, jops, jops_LdL, phase, V, T_block, params = (
        build_ats_hamiltonian_interaction(n_a=n_a, n_b=n_b, alpha_sq=alpha_sq)
    )
    solver = ChebAr(H_I, jops, T_block, jump_ops_LdL=jops_LdL,
                    dims=(n_a, n_b), output_phase=phase)
    x0 = solver.make_x0(seed=0)
    _, _, ritz = solver.first_estimation(x0, m_arnoldi=m_arnoldi_0)
    solver.setup_chebyshev(ritz, margin=margin)
    Q, H, mu_list = solver.arnoldi_hessenberg(x0, solver.chebyshev_filter, m_arnoldi)
    x_ritz, _ = solver.ritz_vector(Q, H, m_arnoldi)
    return to_jsonable({
        "rate_bf": solver.rate_from_mu(mu_list[-1]),
        "x_ritz": x_ritz,
        "res": solver.residual_check(x_ritz),
        "params": params,
        "V": V,
    })
```

`to_jsonable` (already in [`io.py`](../src/floquet_lindblad/io.py)) does double duty here:
the serialization layer written for JSON files is exactly the right cloudpickle
boundary. Nice accident of the existing design.

Client side:

```python
from anb_compute import ray as acr

out = acr.run(
    run_point, 8.5,
    num_gpus=1, num_cpus=8,
    runtime_env=acr.build_runtime_env(
        py_modules=[
            str(REPO / "src" / "floquet_lindblad"),   # by PATH, not by name
            DYNAMIQS,                        # the patched checkout
        ],
    ),
)
```

Caveats from the docs:

- `py_modules` **by path**. A bare `"floquet_lindblad"` resolving to a non-editable
  `site-packages` install ships a frozen copy and your edits never leave the
  container (`build_runtime_env` warns, but quietly).
- `working_dir` is **reserved** by `acr.run`. Read inputs from GCS instead —
  relevant to `--warm-start-file` (§5).
- Logs print when the job **ends**. Follow a long Arnoldi with
  `ray job logs --follow <job_id>`.
- `poll_seconds` defaults to 10 and sets a floor on round-trip time; drop it to
  2 while iterating.
- Each `acr.run` pays one job's startup (measured: ~10–20 s warm). Fine for a
  minutes-long solve, wrong for a loop of small ones.

### DONE: ported and verified

`notebooks/solvers/cheb_ar.ipynb` is now 6 cells: prerequisites (markdown), auth/version
guard + `RUNTIME_ENV`, the `run_cheb_ar` function, the `acr.run` call, the
scalars, and a local matplotlib Wigner plot.

The worker returns **plain numpy/python only**. This is not stylistic: the client
env has neither jax nor dynamiqs, so a returned jax array or `QArray` would fail
to unpickle here. That is also why the Wigner grid is computed on the worker
(`dq.wigner` → `(xvec, yvec, W)` arrays) and the notebook only draws it.

Full run, all six cells green, defaults `n_a=25, n_b=11, alpha_sq=8.5,
m_arnoldi_0=60, m_arnoldi=120, margin=1e-2`:

```
python 3.13.14 | ray 2.56.0        device: cuda:0
timings (s): build 8.1 | first_estimation 5.5 | arnoldi 103.8 | total 119.9
wall clock incl. job startup: 140.2 s

rate_bf : 3.999306340632297e-07
mu      : 0.9999999008303276
mu_RQ   : 0.9999999032686184
rate_RQ : 3.9009751466087275e-07
res_rel : 2.5623470732970314e-09
```

!!! warning "`rate_bf` and `rate_RQ` differ by 2.5% — that is conditioning, not a bug"
    `res_rel ~ 2.6e-9` says the Ritz vector is well converged, and `mu` and
    `mu_RQ` agree to ~2.4e-9. But `mu` sits within `1e-7` of 1, and the rate is
    `-log(mu)/T_block`, so it depends on `1 - mu ~ 9.9e-8`. An absolute agreement
    of 2.4e-9 in `mu` is therefore a ~2.5% spread in the rate. That ratio is
    exactly what the two rates show. Treat ~2–3% as the current precision floor
    on the bit-flip rate at this `m_arnoldi`, and do not read more digits than
    that into either number.

Arnoldi dominates (104 s of 120 s), so the driver-side overheads (~20 s job
startup) are not worth optimizing; widening the fan-out is (§5).

## 5. The sweeps: one job, many tasks

!!! warning "The two sweeps do **not** parallelize the same way"
    This is the one thing to get right before touching either of them.

    **`sweep_alpha.py` fans out.** `eps_p` and `kappa_b` are fixed along it, so
    the interaction-frame eigenbasis never moves and no point needs the previous
    one's answer. One GPU task per point, all submitted at once.

    **`sweep_eps_p.py` is sequential by construction.** Each point is
    warm-started from the previous point's Ritz vector, re-expressed in the new
    eigenbasis via `transform_vectorized_state`, because the basis *does* move
    with `eps_p`. Only the first point runs cold, with the larger
    `--m-arnoldi-first`; the rest reuse the chain with a smaller `--m-arnoldi`.
    Fanning that out silently changes the numerics — every point would run cold.

    So the eps_p driver runs one GPU task per point *in order*, passing `x_ritz`
    and `V` along. `--no-warm-start` breaks the chain **deliberately**: all
    points cold and independent, submitted at once. More GPU-seconds, far less
    wall clock. That is a choice, not a default.

    The same distinction is why `solve_point` has a `want_V` flag: only the
    sequential chain needs the eigenbasis back.

### Failure isolation, and why not `acr.as_completed`

Two independent guards, because a sweep losing every finished point to one bad
point would be the worst outcome:

* `solve_point_safe` **returns** a numerical failure as an error entry instead of
  raising it — the behaviour the pre-cluster scripts had.
* The drivers consume with `ray.wait` / `ray.get` in a try/except rather than
  `acr.as_completed`. `as_completed` calls `ray.get` *inside a generator*, so a
  task lost to infrastructure (node preempted, OOM, retries exhausted) raises
  out of the loop and discards every point already collected.

In the sequential chain a failed point additionally sets `prev = None`, so the
next point restarts cold rather than warm-starting from nothing.

### Verified on the cluster

A reduced smoke run (`n_a=15, n_b=6`) exercised all three paths:

```
fan-out (3 independent points)     all returned rates
chain, point 2 fails ellipse fit   recorded, point 3 restarts warm=False
failure path (n_a=0)               error entry returned, coords preserved
```

The mid-chain failure was the tiny truncation, not a defect: the ladder
escalated `m_arnoldi_0` through 40/57/80 and then halved the margin to 1.95e-5
before giving up. It did usefully prove the chain-restart branch. `escalating_setup`
also has unit tests covering every rung and both failure exits.

Because that run's middle point failed, it never exercised a *successful* warm
start. A second run with closely spaced points (`eps_p = 0.10, 0.12, 0.14`) did:

| eps_p | warm_start | rate | res_rel | s |
| --- | --- | --- | --- | --- |
| 0.10 | False | 3.172228e-04 | 3.09e-04 | 32.7 |
| 0.12 | **True** | 3.972402e-04 | 2.25e-08 | 32.7 |
| 0.14 | **True** | 4.881798e-04 | 6.57e-10 | 34.5 |
| 0.14 | *(cold, cross-check)* | 4.882172e-04 | 1.21e-06 | 33.4 |

Two things worth noting. The warm and cold answers for the same point agree to
**7.7e-5** relative, which is the real check on the `V_prev` basis rotation — a
wrong rotation would not land there. And at equal Krylov size the warm start
converges roughly **1800x** tighter (`res_rel` 6.6e-10 vs 1.2e-6) for the same
~33 s, which is the whole reason the chain exists. The production sweep spends
that margin differently: it drops warm points to a smaller `--m-arnoldi` and
takes the time saving instead.

!!! warning "The full sweeps have not been run"
    Only the reduced smoke run above. The defaults in the drivers and notebooks
    (`n_a=30, n_b=11` and `n_a=32, n_b=12`) are untested end to end and will take
    minutes per point.

[`scripts/cluster/*.sh`](../scripts/cluster/) were OAR files (`#OAR -l
gpu=1,walltime=0:35:00`, `#OAR -p chuc`, `source ~/venvs/dq_gpu/bin/activate`).
All of it is dead here: no queue, no walltime, no partition, no venv. They are
replaced by [`scripts/cluster/submit.py`](../scripts/cluster/submit.py).

The structural point (`ray_scheduling.md`, "Coming from Slurm"): one job per
parameter point is the reflex that backfires. Each `ray job submit` starts a
driver — by default on the shared 4-CPU head — and buys no extra capacity.
**Submit one job, fan out inside it.**

[`scripts/sweep_alpha.py`](../scripts/sweep_alpha.py) is already shaped for this:
`run_for_alphasq(alpha_sq, x0)` is the natural task body. The `for` loop becomes:

```python
refs = [acr.submit(run_for_alphasq, a, num_gpus=1, num_cpus=8)
        for a in args.alpha_sq]
for ref, result in acr.as_completed(refs):
    ...
```

submitted once with `acr.submit_job("python -m ...", entrypoint_num_cpus=1)` —
the `1` matters, it moves the driver off the head onto a worker. Keep per-task
`num_cpus` ≤ 28 so driver and task still fit on one node.

Two things to watch:

- **Retries.** Ray retries a task up to 3× when a node dies. The existing
  per-point `try/except` is unaffected (your own exceptions are not retried),
  and the code accumulates into a list and writes once at the end, so it is
  already idempotent. Keep it that way.
- **Warm starts.** `--warm-start-file` reads a previous results JSON from a local
  path that does not exist on the cluster. It has to come from GCS —
  `acr.load_output(uri)` or `fsspec.open("gs://...")` inside the job.

## 6. Results: GCS, not a local path

All three scripts end with `os.makedirs` + `json.dump` to `args.output`, and the
OAR wrappers write `results/..._$(date +%Y_%m_%d).json`. **The job's disk is
destroyed when the job ends.** Those files are gone.

Use `acr.save_output(local, uri)` to a `gs://anb-ray-results/<prefix>/` you name,
and `acr.load_output(uri, token=acr.gcs_credentials())` to read it back. The
`token=` is required from anb-dev; on JupyterHub it is not.

Same prefix **overwrites silently**, so keep the date/run-id habit the OAR
scripts already had.

## 7. What does not need to change

Worth stating, since the list above is long:

- `src/floquet_lindblad/solvers/cheb_ar.py` — numerics untouched. Model-agnostic, pure
  Python, double precision, jits fine on the image's jax. Only the
  `require_gpu` guard is cluster-relevant, and it stays.
- `src/floquet_lindblad/models/ats.py` — physics constants and the three frame builders
  unaffected.
- `src/floquet_lindblad/io.py` — unaffected, and doubles as the task-result boundary.
- Module-level `jax_enable_x64` / `dq.set_precision("double")` — correct as-is;
  they run on the worker at import, and x64 was verified working there.
- The `jax`/`jaxlib`/`diffrax` pins in `pyproject.toml` — they match the image
  exactly. Only the `dynamiqs` line needs revisiting (§2).

## Suggested order

Done:

1. ~~Resolve `mesolve_fast`~~ (§2) — rebased onto v0.3.6, pushed, pin updated.
2. ~~Port `notebooks/solvers/cheb_ar.ipynb` to a single `acr.run` function~~ (§4).
3. ~~Strip `--gpu-id` / `CUDA_VISIBLE_DEVICES`~~ from the three scripts (§3).
4. ~~Delete `scripts/cluster/*.sh`~~ (the OAR job files).

Next:

5. Port `sweep_alpha.py` to `submit_job` + `acr.submit`/`as_completed` +
   `save_output` (§5, §6); `sweep_eps_p.py` follows the same shape.
6. Port the three remaining test notebooks (`test_cheb_ar_bis`,
   `test_sweep_alpha`, `test_sweep_eps_p`) the same way as §4.
7. Consider upstreaming the `mesolve_fast` patch, which would remove the
   `py_modules` dynamiqs entry entirely (§2).

## Appendix: client setup (anb-dev)

`pip` is `pixi` here. The client must match the cluster: **ray 2.56.0, Python
3.13.x**. `/workspace/pixi.toml` is set up for this (`python ==3.13.14`, plus
`anb-compute` from the internal GitLab index, which authenticates via the mounted
`~/.netrc`).

Per shell that submits:

```bash
export VAULT_ADDR=https://vault.int.alice-bob.com
vault login -method=oidc role=research_core_user   # 8h–7d TTL
export RAY_AUTH_MODE=token
export RAY_AUTH_TOKEN=$(vault kv get -field=token kv-it/prod/ray/auth-token)
```

Both variables, before the kernel starts — Ray reads `RAY_AUTH_MODE` once at
import and caches it; setting it later gets you a 401.

Not in the setup docs: `acr.run` also needs **GCS read credentials** to fetch its
result envelope, via `acr.gcs_credentials()` → `vault read
gcp/impersonated-account/ray-results-reader/token`. On first use this triggers a
separate interactive Microsoft device-code login. Expect it; it is cached
afterwards.
