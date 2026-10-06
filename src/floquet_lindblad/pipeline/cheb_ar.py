"""One sweep point with ``ChebAr``: build -> filter -> Arnoldi -> rate -> residual.

The secondary pipeline; the main one is
:mod:`floquet_lindblad.pipeline.krylov_schur`. The two share their calling
contract (sweep coordinates, ``x0`` / ``V_prev`` warm start, ``want_x_ritz`` /
``want_V``), so a sweep driver can switch between them.

This used to be copy-pasted four times, in near-identical form, across
``scripts/sweep_alpha.py``, ``scripts/sweep_eps_p.py``,
``notebooks/sweeps/legacy_cheb_ar/sweep_alpha.ipynb`` and ``notebooks/sweeps/legacy_cheb_ar/sweep_eps_p.ipynb``. The two
sweeps differ only in *what* varies along them (``alpha_sq`` at fixed basis, or
``eps_p`` with a basis that moves and so needs
:func:`~floquet_lindblad.models.base.transform_vectorized_state`), never in the pipeline
itself.

It lives in the package rather than in the scripts because cluster workers
import it: the sweep drivers ship ``floquet_lindblad`` through the Ray ``runtime_env``
and each task calls :func:`solve_point_cheb_ar`.

Everything :func:`solve_point_cheb_ar` returns is plain numpy / python — no jax arrays,
no dynamiqs objects — so a result can cross a process boundary (cloudpickle to
a Ray driver, JSON to disk, a notebook client that has neither library
installed) without the receiver needing the GPU stack.
"""

import numpy as np

from floquet_lindblad.pipeline.common import error_entry

DEFAULT_MIN_MARGIN = 1e-5


def escalating_setup(
    solver,
    x0,
    m_arnoldi_0,
    margin,
    *,
    min_margin=DEFAULT_MIN_MARGIN,
    log=print,
):
    """First estimation + Chebyshev ellipse fit, with the escalation ladder.

    The ellipse fit can fail when the estimated spectrum is too poorly resolved
    to enclose. The ladder, unchanged from the original scripts:

    1. try ``m_arnoldi_0``, then ``m_arnoldi_0 * sqrt(2)``, then
       ``2 * m_arnoldi_0`` — each a fresh unfiltered estimation;
    2. if the fit still fails, keep the *last* estimation and halve ``margin``
       down to ``min_margin``.

    Parameters
    ----------
    solver : ChebAr
        Mutated in place: on success its Chebyshev filter is set up.
    x0 : array
        Starting vector for the unfiltered Arnoldi.
    log : callable
        Where the per-attempt failure notes go.

    Returns
    -------
    (m_arnoldi_0_used, margin_used) : tuple[int, float]

    Raises
    ------
    Exception
        The last failure, if no rung of the ladder succeeds.
    """
    m_schedule = [
        m_arnoldi_0,
        int(round(m_arnoldi_0 * np.sqrt(2))),
        2 * m_arnoldi_0,
    ]
    ritz_vals = None
    last_exc = None

    for m0 in m_schedule:
        try:
            _, _, ritz_vals = solver.first_estimation(x0, m_arnoldi=m0)
            solver.setup_chebyshev(ritz_vals, margin=margin)
            return m0, margin
        except Exception as exc:  # noqa: BLE001 - the ladder is the handler
            last_exc = exc
            log(f"setup_chebyshev failed (m_arnoldi_0={m0}, margin={margin:.3e}): {exc}")

    if ritz_vals is None:
        raise last_exc  # even the unfiltered estimation itself failed

    # Keep the widest estimation, shrink the margin instead.
    trial = margin / 2
    while trial >= min_margin:
        try:
            solver.setup_chebyshev(ritz_vals, margin=trial)
            return m_schedule[-1], trial
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log(
                f"setup_chebyshev failed "
                f"(m_arnoldi_0={m_schedule[-1]}, margin={trial:.3e}): {exc}"
            )
            trial /= 2

    raise last_exc


def solve_point_cheb_ar(
    *,
    n_a,
    n_b,
    alpha_sq,
    eps_p,
    kappa_b=None,
    cheb_degree=6,
    m_arnoldi_0=60,
    m_arnoldi=80,
    margin=5e-3,
    seed=0,
    x0=None,
    V_prev=None,
    min_margin=DEFAULT_MIN_MARGIN,
    want_x_ritz=True,
    want_V=False,
    want_wigner=False,
    log=print,
):
    """Run the interaction-frame ChebAr pipeline for one parameter point.

    Parameters
    ----------
    kappa_b : float, optional
        Buffer loss rate; the model default (``ats.KAPPA_B``) when omitted.
    x0 : array, optional
        Warm-start vector. When ``V_prev`` is also given, it is first
        re-expressed in this point's eigenbasis (needed for an ``eps_p`` sweep,
        where the basis moves; a no-op for an ``alpha_sq`` sweep).
    want_x_ritz, want_V : bool
        Include the Ritz vector / frame transform in the result. Both are
        ``N**2`` and ``N**2``-sized complex arrays (tens of MB for a big
        truncation), so leave them off when the caller only wants the rate.
        ``want_V`` is what a *sequential* warm-started sweep needs.
    want_wigner : bool
        Also return the storage-mode Wigner grid, computed here so a client
        without dynamiqs can plot it.

    Returns
    -------
    dict
        Plain numpy / python only. Always carries the sweep coordinates, the
        settings actually used (``m_arnoldi_0``, ``margin`` — which may differ
        from the requested ones, see :func:`escalating_setup`), ``rate_bf`` and
        ``res``.
    """
    import time

    from floquet_lindblad.models.ats import KAPPA_B, build_ats_hamiltonian_interaction
    from floquet_lindblad.models.base import transform_vectorized_state
    from floquet_lindblad.solvers.cheb_ar import ChebAr

    def py(v):
        """0-d jax/numpy scalar -> plain python float/complex."""
        a = np.asarray(v)
        return complex(a) if np.iscomplexobj(a) else float(a)

    t0 = time.time()
    kappa_b = KAPPA_B if kappa_b is None else kappa_b

    (
        H_I,
        jump_ops_I,
        jump_ops_LdL_I,
        output_phase,
        V,
        T_block,
        params,
    ) = build_ats_hamiltonian_interaction(
        n_a=n_a,
        n_b=n_b,
        alpha_sq=alpha_sq,
        kappa_b=kappa_b,
        epsilon_p=eps_p,
    )

    solver = ChebAr(
        H_I,
        jump_ops_I,
        T_block,
        jump_ops_LdL=jump_ops_LdL_I,
        output_phase=output_phase,
        dims=(n_a, n_b),
        cheb_degree=cheb_degree,
    )

    if x0 is None:
        x0 = solver.make_x0(seed=seed)
        warm_start = False
    else:
        if V_prev is not None:
            # The interaction-frame basis moves with eps_p, so a warm vector
            # from the previous point has to be rotated into this one.
            x0 = transform_vectorized_state(x0, np.asarray(V_prev), np.asarray(V))
        warm_start = True

    m_arnoldi_0_used, margin_used = escalating_setup(
        solver, x0, m_arnoldi_0, margin, min_margin=min_margin, log=log
    )
    t_setup = time.time() - t0

    Q, H, mu_list = solver.arnoldi_hessenberg(
        x0, solver.chebyshev_filter, m_arnoldi, warm_start=warm_start
    )
    rate_bf = solver.rate_from_mu(mu_list[-1])
    x_ritz, rho_ritz = solver.ritz_vector(Q, H, m_arnoldi, target=mu_list[-1])
    res = solver.residual_check(x_ritz)

    out = {
        # sweep coordinates
        "alpha_sq": float(alpha_sq),
        "eps_p": float(eps_p),
        "kappa_b": float(kappa_b),
        "n_a": int(n_a),
        "n_b": int(n_b),
        # settings actually used (the ladder may have moved them)
        "cheb_degree": int(cheb_degree),
        "m_arnoldi_0": int(m_arnoldi_0_used),
        "m_arnoldi": int(m_arnoldi),
        "margin": float(margin_used),
        "warm_start": bool(warm_start),
        # results
        "rate_bf": py(rate_bf),
        "mu": py(mu_list[-1]),
        "mu_list": np.asarray(mu_list),
        "res": {k: py(v) for k, v in res.items()},
        "params": {k: py(v) for k, v in params.items()},
        "timings_s": {"setup": t_setup, "total": time.time() - t0},
    }
    if want_x_ritz:
        out["x_ritz"] = np.asarray(x_ritz)
    if want_V:
        out["V"] = np.asarray(V)
    if want_wigner:
        import dynamiqs as dq

        xvec, yvec, W = dq.wigner(dq.ptrace(rho_ritz, 0))
        out["wigner"] = (np.asarray(xvec), np.asarray(yvec), np.asarray(W))
    return out


#: Keys copied onto an `error_entry` so a failed point still says where it was.
_COORD_KEYS = (
    "alpha_sq", "eps_p", "kappa_b", "n_a", "n_b",
    "cheb_degree", "m_arnoldi_0", "m_arnoldi", "margin",
)


def solve_point_cheb_ar_safe(**kwargs):
    """`solve_point_cheb_ar`, returning a failure instead of raising it.

    This is what a sweep should submit as its Ray task. Two reasons the
    exception must not escape:

    * a single unconvergeable point should be *recorded* and the sweep carry on,
      which is what the pre-cluster scripts did;
    * a raising task makes ``ray.get`` raise, which aborts the driver's
      ``as_completed`` loop and discards every point that already finished.

    Genuine infrastructure failures (node preempted, OOM) still raise, and are
    still retried by Ray; the driver handles those separately.
    """
    try:
        return solve_point_cheb_ar(**kwargs)
    except Exception as exc:  # noqa: BLE001 - one bad point must not sink the sweep
        return error_entry(exc, **{k: kwargs.get(k) for k in _COORD_KEYS})
