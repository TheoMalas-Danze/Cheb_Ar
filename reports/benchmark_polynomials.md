# Benchmark: polynomial filters in U -- U^n vs T_n(U)

> **Cat-size convention.** The runs below predate the 2026-09 fix of
> `epsilon_d` (see `docs/models_ats.md` §2): here `alpha_sq` is twice the
> cat size, `|alpha|^2 ~ alpha_sq / 2`. The same physical points are now
> `alpha_sq` = half the values quoted.

Sources: `notebooks/benchmarks/benchmark_polynomials.ipynb` (`alpha_sq = 6`) and
`notebooks/benchmarks/benchmark_polynomials_small_cat.ipynb` (`alpha_sq = 3`). Same code, only
`alpha_sq` differs.

Both notebooks compare two degree-`n` polynomials of the one-drive-period
propagator `U`, for `n = 1..10`, each run through a fixed `m = 60` Arnoldi
steps from the same seed:

| key | polynomial | realization |
|---|---|---|
| `power` | `U^n` | `ArnoldiLindblad` on an `n`-period block (`n_periods = n`) |
| `cheb` | `T_n(U)` | `ChebAr` on a 1-period block with `cheb_degree = n` (`m_cheb_0 = 60`, `margin = 1e-2`) |

Other parameters are those of `notebooks/benchmarks/benchmark_methods.ipynb`: `n_a = 25`,
`n_b = 11`, `epsilon_p = 0.1`, `seed = 0`. Cost is counted in base
(single-period) propagator applications: `apps_power = 60 n`,
`apps_cheb = 60 + 60 n` (the extra 60 is `ChebAr`'s one-off unfiltered
estimation used to fit the ellipse). The reference rate is `power` at
`n = 10`, the most-converged block available. Everything ran in one job on
one GPU per `alpha_sq`.

## alpha_sq = 6 -- reference `rate_RQ = 9.908148e-06`

| n | power rate_bf | rel. dev. | res_rel | run [s] | cheb rate_bf | rel. dev. | res_rel | run [s] |
|---|---|---|---|---|---|---|---|---|
| 1 | 9.440e-02 | 9.5e+03 | 1.3e-01 | 5.5 | 9.440e-02 | 9.5e+03 | 1.3e-01 | 7.9 |
| 2 | 1.763e-03 | 1.8e+02 | 7.7e-03 | 8.4 | 1.187e-03 | 1.2e+02 | 4.2e-03 | 8.4 |
| 3 | 2.551e-04 | 2.5e+01 | 1.4e-03 | 10.6 | 3.287e-04 | 3.2e+01 | 3.8e-03 | 11.8 |
| 4 | 4.674e-04 | 4.6e+01 | 4.4e-03 | 12.9 | 3.945e-05 | 3.0e+00 | 1.4e-02 | 15.0 |
| 5 | 1.461e-05 | 4.7e-01 | 6.4e-04 | 16.2 | 8.322e-06 | 1.6e-01 | 1.5e-04 | 18.1 |
| 6 | 9.311e-06 | 6.0e-02 | 3.5e-05 | 18.0 | 1.003e-05 | 1.3e-02 | 1.5e-05 | 21.2 |
| 7 | 9.686e-06 | 2.2e-02 | 2.9e-06 | 21.0 | 9.836e-06 | 7.3e-03 | 1.5e-05 | 24.1 |
| 8 | 9.556e-06 | 3.6e-02 | 1.1e-05 | 23.5 | 9.980e-06 | 7.3e-03 | 2.7e-06 | 27.2 |
| 9 | 9.870e-06 | 3.9e-03 | 2.0e-06 | 25.9 | 9.908e-06 | 1.2e-05 | 4.3e-08 | 30.7 |
| 10 | 9.910e-06 | 1.5e-04 | 4.6e-08 | 28.0 | 9.908e-06 | 2.4e-05 | 2.3e-09 | 33.2 |

## alpha_sq = 3 -- reference `rate_RQ = 4.361288e-04`

| n | power rate_bf | rel. dev. | res_rel | run [s] | cheb rate_bf | rel. dev. | res_rel | run [s] |
|---|---|---|---|---|---|---|---|---|
| 1 | 8.629e-02 | 2.0e+02 | 1.2e-01 | 5.8 | 8.629e-02 | 2.0e+02 | 1.2e-01 | 8.2 |
| 2 | 2.505e-03 | 4.7e+00 | 1.7e-02 | 8.0 | 8.110e-04 | 8.6e-01 | 9.1e-03 | 7.9 |
| 3 | 6.888e-05 | 8.4e-01 | 2.4e-03 | 9.4 | 1.949e-04 | 5.5e-01 | 4.8e-03 | 10.9 |
| 4 | 2.633e-04 | 4.0e-01 | 7.0e-03 | 11.8 | 5.129e-04 | 1.8e-01 | 1.5e-02 | 13.7 |
| 5 | 4.744e-04 | 8.8e-02 | 5.7e-03 | 14.2 | 3.010e-04 | 3.1e-01 | 6.4e-04 | 16.3 |
| 6 | 2.902e-04 | 3.4e-01 | 1.3e-03 | 16.1 | 1.653e-04 | 6.2e-01 | 1.2e-02 | 18.9 |
| 7 | 2.564e-04 | 4.1e-01 | 7.3e-03 | 18.1 | 3.053e-04 | 3.0e-01 | 1.1e-03 | 22.2 |
| 8 | 6.088e-04 | 4.0e-01 | 6.1e-03 | 20.9 | 2.677e-04 | 3.9e-01 | 4.2e-05 | 24.2 |
| 9 | 4.475e-04 | 2.6e-02 | 8.5e-04 | 22.2 | 2.680e-04 | 3.9e-01 | 4.1e-06 | 26.7 |
| 10 | 4.361e-04 | 3.9e-06 | 1.0e-04 | 24.2 | 2.681e-04 | 3.9e-01 | 1.3e-07 | 29.6 |

## Timings

- `t_apply` for `power` grows linearly with `n` (`~110 ms` at `n = 1` to
  `~0.9-1.0 s` at `n = 10`), as expected for an `n`-period `mesolve`.
  `t_apply` for `cheb` is the single-period call, flat at `~105-120 ms`;
  the filter itself applies it `n` times.
- Total Arnoldi run time is therefore nearly identical between the two
  methods at equal `n` (e.g. `n = 10`: 28.0 s vs 33.2 s at `alpha_sq = 6`).
  `cheb` is consistently ~2-5 s slower, which is the 60 extra unfiltered
  applications of `m_cheb_0` plus the Chebyshev recurrence overhead. Equal
  `n` really is equal cost here, so the comparison below is a fair one.

## Observations

**alpha_sq = 6: both work, and `T_n(U)` converges about one degree earlier
than `U^n`.**

- `n = 1` is identical for both (a degree-1 Chebyshev filter is an affine
  map of `U`, so the Krylov space is the same) and completely wrong
  (`rate ~ 0.09`, four orders of magnitude off): 60 Arnoldi steps on `U`
  itself do not resolve an eigenvalue this close to 1.
- Neither method is usable below `n = 5`; both deliver at `n = 6`, but
  with different quality: `power` at `n = 6` is 6% off (`res_rel 3.5e-5`),
  `cheb` at `n = 6` is 1.3% off (`res_rel 1.5e-5`).
- From `n = 6` on, `cheb`'s relative deviation is lower than `power`'s at
  every `n` (by 2x to 300x), and at `n = 9-10` it is already at the
  `1e-5` level with `res_rel ~ 1e-8-1e-9`, i.e. at the integrator noise
  floor. `power` needs `n = 10` to reach `1.5e-4`.
- On the cost axis the picture is the same: `cheb` at `n = 9`
  (600 apps) is more accurate than `power` at `n = 10` (600 apps).
- The `power` curve is not monotone (`n = 4` is worse than `n = 3`,
  `n = 8` worse than `n = 7`): with a fixed `m = 60` the Ritz value the
  tracker locks onto still jumps between candidates until the gap is large
  enough.

**alpha_sq = 3: `T_n(U)` converges to the wrong eigenvalue, and looks
converged while doing it.**

- `power` behaves as at `alpha_sq = 6`, only slower: it is noisy up to
  `n = 8` (deviations of 30-40%, non-monotone) and then locks in at
  `n = 9-10` (2.6% then `4e-6`). `n = 10` is the only really converged
  `power` point, which is why it is the reference.
- `cheb` never gets there. From `n = 8` on it is fully self-consistent --
  `rate_bf = 2.68e-4` to four digits, `res_rel` dropping from `4e-5` to
  `1.3e-7` -- and 38.5% below the reference, at every `n`. This is the
  same `2.68e-4` that `notebooks/benchmarks/benchmark_methods.ipynb` found for `cheb_ar` at
  `alpha_sq = 3` with `m = 90`, `cheb_degree = 6`. Adding degree does not
  fix it: the filter is converging cleanly to an eigenvalue of `T_n(U)` that
  is not the one we want.
- The most likely mechanism is the ellipse fit. `setup_chebyshev` takes the
  largest-real-part Ritz value of the 60-step unfiltered estimation as the
  target and encloses the rest. At `alpha_sq = 3` the spectrum is denser
  near 1 (the target rate is ~50x larger than at `alpha_sq = 6`, so the
  gap to the next eigenvalues is smaller in relative terms), and the same
  `m_cheb_0 = 60` estimation that resolves the target at `alpha_sq = 6`
  evidently does not here: either the true target ends up inside the
  ellipse (and is damped), or a different eigenvalue ends up outside it
  (and is amplified). `power` is immune to this because it has no
  estimation step -- it just needs enough `n`.
- Again, `res_rel` is useless as a correctness check for `cheb`: at
  `alpha_sq = 3`, `n = 10`, it is `1.3e-7`, three orders of magnitude
  better than the correct `power` answer at the same `n` (`1.0e-4`).

## Takeaway

At equal degree the two polynomials cost the same, and where the Chebyshev
filter is set up correctly (`alpha_sq = 6`) `T_n(U)` beats `U^n` by roughly
one degree, or 10-300x in accuracy at the same `n` -- the expected
advantage, though smaller than the exponential-vs-geometric argument would
suggest, because with `m = 60` the cost is dominated by the fixed Arnoldi
depth rather than by `n`.

At `alpha_sq = 3` the advantage is moot: `T_n(U)` converges, with excellent
residuals, to an eigenvalue 38.5% below the correct one, at every `n >= 8`.
Increasing `cheb_degree` does not fix this, so the defect is upstream of the
filter degree -- in `first_estimation`/`setup_chebyshev` (`m_cheb_0`,
`margin`, the ellipse fit) -- and that is where the next investigation
should go. Concretely: rerun `alpha_sq = 3` with a larger `m_cheb_0`
(e.g. 90-120) and/or a smaller `margin`, and inspect the ellipse against
the `power n = 10` Ritz spectrum to see which side of the vertex the true
target lands on.

Until then, `U^n` with `n = 10` (a 10-period block) is the safe default at
both `alpha_sq`; it is only ~15% slower than `T_n(U)` at the same `n` and
cannot be misled by a bad spectrum estimate.
