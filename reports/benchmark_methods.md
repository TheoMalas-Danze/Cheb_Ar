# Benchmark: ChebAr vs plain Arnoldi vs naive Arnoldi

Source: `tests/benchmark_methods.ipynb`. Three ways to estimate the bit-flip
rate (decay rate of the slowest non-trivial eigenvalue of the
trace-projected Floquet propagator), compared at `alpha_sq = 3` and
`alpha_sq = 6` on identical Hilbert spaces and the same GPU (one job per
`alpha_sq`, so timings within a job are directly comparable).

| key | solver | block | m | notes |
|---|---|---|---|---|
| `cheb_ar` | `ChebAr` | 6 drive periods | 90 (`cheb_degree=6`, `margin=1e-2`) | Chebyshev-filtered Arnoldi |
| `arnoldi` | `ArnoldiLindblad` | 6 drive periods | 90 | unfiltered, used as reference |
| `arnoldi_naiv` | `ArnoldiLindblad` | 1 drive period | 540 (`= 90 * 6`) | unfiltered, cheap applications |

All three are compared on the predicted bit-flip rate
`rate_k = -log(mu_k) / T_block`, which absorbs the 6-vs-1-period rescaling of
`mu`, plotted against propagator applications and against estimated wall
clock (`t_apply * apps`).

## Converged answers (final step of each run)

**alpha_sq = 3.0** — reference: `arnoldi`, `rate_RQ = 4.347973e-04`

| method | m | apps | rate_bf | rel. dev. | res_rel | run [s] |
|---|---|---|---|---|---|---|
| cheb_ar | 90 | 600 | 2.680111e-04 | 3.84e-01 | 1.15e-06 | 32.1 |
| arnoldi | 90 | 90 | 4.348001e-04 | 6.44e-06 | 1.11e-04 | 23.4 |
| arnoldi_naiv | 540 | 540 | 4.391344e-04 | 9.97e-03 | 1.98e-08 | 199.0 |

**alpha_sq = 6.0** — reference: `arnoldi`, `rate_RQ = 9.910757e-06`

| method | m | apps | rate_bf | rel. dev. | res_rel | run [s] |
|---|---|---|---|---|---|---|
| cheb_ar | 90 | 600 | 9.907873e-06 | 2.91e-04 | 7.47e-09 | 35.1 |
| arnoldi | 90 | 90 | 9.913605e-06 | 2.87e-04 | 8.73e-08 | 26.5 |
| arnoldi_naiv | 540 | 540 | 9.907906e-06 | 2.88e-04 | 2.00e-09 | 192.1 |

## Measured timings (m = 90 / 540)

| method | alpha_sq=3 build/setup/run [s] | t_apply | alpha_sq=6 build/setup/run [s] | t_apply |
|---|---|---|---|---|
| cheb_ar | 8.9 / 4.9 / 32.1 | 112.3 ms | 8.5 / 5.2 / 35.1 | 120.7 ms |
| arnoldi | 0.7 / 0.0 / 23.4 | 548.7 ms | 0.7 / 0.0 / 26.5 | 603.8 ms |
| arnoldi_naiv | 0.4 / 0.0 / 199.0 | 107.2 ms | 0.5 / 0.0 / 192.1 | 121.3 ms |

## Observations

- **With `m` raised to 90 (was 60) and `m_naiv` to 540 (was 360), agreement
  is much better than in the previous run** at `alpha_sq = 6`: all three
  methods now agree with `arnoldi` to ~3e-4, and `res_rel` is at the 1e-6 to
  1e-9 level across the board — this looks converged.
- **At `alpha_sq = 3`, `cheb_ar` is still the outlier**: it disagrees with
  the `arnoldi`/`arnoldi_naiv` consensus (which agree with each other to
  ~1e-2) by 38%, despite its own `res_rel` being tiny (1.15e-6). This
  confirms the earlier finding: a small `res_rel` for `cheb_ar` is not
  sufficient evidence that the filtered Ritz value is correct — the filter
  itself (`margin`, `cheb_degree`, ellipse fit from `first_estimation`) is
  the likely culprit, per `docs/arnoldi_no_cheb.md`'s reading guide.
  `arnoldi` and `arnoldi_naiv` now agree with each other at `alpha_sq = 3`
  (9.97e-3 relative deviation), which strengthens confidence in the plain
  Arnoldi reference at this point and further isolates `cheb_ar` as the
  problem.
- On wall clock, `arnoldi` remains fastest at `alpha_sq = 6` (26.5 s) and is
  close behind `cheb_ar` at `alpha_sq = 3` (23.4 s vs. 32.1 s) while
  actually converging correctly there. `arnoldi_naiv` is far more expensive
  in wall clock (192–199 s) than either 6-period method, despite its cheap
  individual applications (~110–120 ms, similar to `cheb_ar`'s), because it
  needs 6x more steps to cover the same physical time.
- `t_apply` for `arnoldi` (549–604 ms) is markedly higher than for `cheb_ar`
  and `arnoldi_naiv` (107–121 ms) — as expected, since `cheb_ar`/`arnoldi_naiv`
  measure the cost of one *filtered/short* propagator call, while `arnoldi`'s
  applications integrate a full 6-period block directly (no Chebyshev
  compression of that cost). This is why the application-count axis alone is
  misleading and wall clock should drive the method choice.

## Takeaway

At `m=90` (`arnoldi`), `m=90` with `cheb_degree=6` (`cheb_ar`), and
`m=540` (`arnoldi_naiv`), the plain unfiltered `arnoldi` and `arnoldi_naiv`
now agree with each other at both `alpha_sq` values, and are the trustworthy
answers. `cheb_ar` converges correctly at `alpha_sq = 6` but is still wrong
by ~38% at `alpha_sq = 3` — the Chebyshev filter's parameters (`margin`,
`cheb_degree`, or the ellipse fit driving `setup_chebyshev`) need further
tuning before `cheb_ar` can be trusted uniformly. In terms of speed, plain
`arnoldi` (6-period block) remains the best default: it's fastest or
near-fastest at both points and, unlike `cheb_ar`, gives the right answer at
`alpha_sq = 3` too. `arnoldi_naiv` is not wall-clock competitive.
