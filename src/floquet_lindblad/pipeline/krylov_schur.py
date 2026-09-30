"""One sweep point with ``KrylovSchurLindblad`` — the main pipeline.

Build the interaction-frame Lindbladian with *one* drive period per block, run
the thick-restarted Arnoldi on the one-period propagator ``U`` in two
tolerance phases (relaxed cycles, then polishing cycles at the final tolerance
resuming the kept subspace), and read the rate off a long-block Rayleigh
quotient at tight tolerance. See ``docs/krylov_schur.md``.

This used to be a notebook-level ``solve_point_ks``, copied into the three
Krylov-Schur sweep notebooks; the copy in
``notebooks/sweeps/sweep_alpha_eps_p.ipynb`` was the superset (warm start,
adaptive final block, ``mu_hist``) and is the one kept here. With the default
``n_blocks_auto=False`` it reduces to the ``alpha_sq`` sweep's version.

Same calling contract as :func:`~floquet_lindblad.pipeline.cheb_ar.solve_point_cheb_ar`
(sweep coordinates, ``x0`` / ``V_prev`` warm start, ``want_x_ritz`` /
``want_V``), and the same plain-numpy return, so a result can cross to a client
without the GPU stack. The keys differ where the solvers do: the residual is
``res_rel`` / ``res_rel_n`` here, ``res["res_rel"]`` there; ``rate_bf`` is the
rate in both.
"""

import numpy as np

from floquet_lindblad.pipeline.common import error_entry


def solve_point_ks(
    *,
    alpha_sq,
    eps_p,
    n_a,
    n_b,
    kappa_b=None,
    # Krylov-Schur
    m=50,
    k=12,
    res_tol=1e-10,
    max_cycles=35,
    # integrator tolerance for the Arnoldi loop ...
    rtol_loop=1e-9,
    atol_loop=1e-10,
    # ... switched to the final tolerance once the residual reaches this, or
    # after `max_cycles_loop` cycles (the relaxed tolerance has a residual
    # floor of its own, which `res_tol_loop` may sit below) ...
    res_tol_loop=1e-8,
    max_cycles_loop=None,
    # ... which is also what the final Rayleigh quotient uses
    rtol_final=1e-11,
    atol_final=1e-12,
    n_blocks_final=50,
    n_blocks_auto=False,
    rq_resolution=1e-5,
    # warm start
    x0=None,
    V_prev=None,
    want_x_ritz=False,
    want_V=False,
    seed=0,
    k_top=3,
):
    """Run the interaction-frame Krylov-Schur pipeline for one parameter point.

    The defaults are band 3 of the ``alpha_sq`` ladder in
    ``notebooks/sweeps/sweep_alpha.ipynb`` (``alpha_sq < 6.5``); the sweeps pass
    their own settings per point.

    Parameters
    ----------
    kappa_b : float, optional
        Buffer loss rate; the model default (``ats.KAPPA_B``) when omitted.
        The ``eps_p`` sweeps rescale it along the sweep.
    n_blocks_final, n_blocks_auto, rq_resolution
        Length of the final Rayleigh-quotient block. Fixed at
        ``n_blocks_final`` by default; with ``n_blocks_auto`` it is shortened
        to what gives ``1 - mu**n ~ rq_resolution``, capped at
        ``n_blocks_final`` (worth it when the gap grows along a sweep, as it
        does with ``eps_p``).
    x0, V_prev : array, optional
        Warm start: the previous point's Ritz vector and the eigenbasis it is
        written in. The interaction-frame basis moves with ``eps_p``, so the
        vector is rotated into this point's basis first. Both solvers
        vectorize column-major, so ``transform_vectorized_state`` applies
        unchanged. Without ``V_prev`` the vector is used as-is (an
        ``alpha_sq`` sweep, where the basis does not move).
    want_x_ritz, want_V : bool
        Ship the Ritz vector / the eigenbasis back, which is what the next
        point of a warm chain needs. Off by default: they are the only large
        objects in the payload.
    k_top : int
        Number of top Fock levels summed into ``tail_a`` / ``tail_b``, the
        truncation check.

    Returns
    -------
    dict
        Plain numpy / python only. Physics constants come from
        ``floquet_lindblad.models.ats`` (single source of truth).
    """
    import time

    import jax

    from floquet_lindblad.models.ats import (
        KAPPA_B,
        build_ats_hamiltonian_interaction,
        transform_vectorized_state,
    )
    from floquet_lindblad.solvers.krylov_schur import KrylovSchurLindblad

    def py(v):
        """0-d jax/numpy scalar -> plain python float/complex."""
        a = np.asarray(v)
        return complex(a) if np.iscomplexobj(a) else float(a)

    lines = []

    def log(s):
        print(s)
        lines.append(s)

    t0 = time.time()
    dims = (n_a, n_b)
    kappa_b = KAPPA_B if kappa_b is None else kappa_b

    # --- interaction frame, ONE drive period per block: the iteration runs on U ---
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
        n_periods=1,
    )

    solver = KrylovSchurLindblad(
        H_I,
        jump_ops_I,
        T_block,
        jump_ops_LdL=jump_ops_LdL_I,
        dims=dims,
        output_phase=output_phase,
        rtol=rtol_loop,
        atol=atol_loop,
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
    t_build = time.time() - t0

    # --- phase 1: cycles at the loop tolerance -------------------------------
    # A warm vector usually clears `res_tol_loop` in the first cycle and falls
    # straight through to the polishing phase; a cold one does not.
    same_tol = (rtol_loop, atol_loop) == (rtol_final, atol_final)
    if max_cycles_loop is None:
        max_cycles_loop = max(1, max_cycles // 2)
    t1 = time.time()
    out = solver.solve(
        x0,
        m=m,
        k=k,
        res_tol=res_tol if same_tol else res_tol_loop,
        max_cycles=max_cycles if same_tol else max_cycles_loop,
        log=log,
    )
    history = list(out["history"])
    n_cycles_loop = out["n_cycles"]
    t_loop = time.time() - t1

    # --- phase 2: polish at the final tolerance, resuming the kept subspace ---
    t2 = time.time()
    if not same_tol:
        log(f"-- switching integrator tolerance to rtol={rtol_final:g}, atol={atol_final:g}")
        solver.set_tolerance(rtol_final, atol_final)
        out = solver.solve(
            state=out["state"],
            m=m,
            k=k,
            res_tol=res_tol,
            max_cycles=max(1, max_cycles - n_cycles_loop),
            log=log,
        )
        history += list(out["history"])
    t_polish = time.time() - t2

    x_ritz = out["x_ritz"]

    # --- how long a final block to use --------------------------------------
    # The rate is read off 1 - mu**n ~ n * (1 - mu), and the relative error of
    # that reading is about rtol_final / (n * (1 - mu)) -- which is why the
    # block is long at all. `1 - mu` *grows* with eps_p, so a block sized for
    # the first point of an eps_p chain is waste at the last: with
    # `n_blocks_auto`, aim for a fixed absolute 1 - mu**n = `rq_resolution`
    # instead, capped at `n_blocks_final`.
    gap = max(float(1 - complex(out["mu"]).real), 1e-14)
    n_blocks = (int(np.clip(int(np.ceil(rq_resolution / gap)), 1, n_blocks_final))
                if n_blocks_auto else int(n_blocks_final))
    log(f"-- final Rayleigh quotient on {n_blocks} block(s): "
        f"1 - mu = {gap:.2e} per period, 1 - mu**n ~ {n_blocks * gap:.2e} "
        f"({n_blocks * gap / rtol_final:.1e} x rtol_final)")

    # --- final numbers on the true propagator, tight tolerance ---------------
    t3 = time.time()
    res_1 = solver.residual_check(x_ritz, n_blocks=1, rtol=rtol_final, atol=atol_final)
    res_n = solver.residual_check(
        x_ritz, n_blocks=n_blocks, rtol=rtol_final, atol=atol_final
    )
    t_final = time.time() - t3

    # --- Fock weights, for the truncation check (lab frame, as in the other
    # notebooks). Only the two marginals travel, not rho itself.
    V = np.asarray(V)
    rho_lab = V @ np.asarray(solver.unvec(x_ritz)) @ V.conj().T
    D = np.einsum("ij,ij->i", rho_lab, rho_lab.conj()).real.reshape(n_a, n_b)
    w_a = D.sum(axis=1) / D.sum()
    w_b = D.sum(axis=0) / D.sum()

    res = {
        # sweep coordinates
        "alpha_sq": float(alpha_sq),
        "eps_p": float(eps_p),
        "kappa_b": float(kappa_b),
        "n_a": int(n_a),
        "n_b": int(n_b),
        "warm_start": bool(warm_start),
        # eigenvalue from the Krylov iteration (one period) ...
        "mu": complex(out["mu"]),
        "rate_ritz": py(solver.rate_from_mu(out["mu"])),
        "res_rel": float(out["res_rel"]),
        "converged": bool(out["converged"]),
        # ... and from the final Rayleigh quotients at tight tolerance
        "mu_RQ_1": py(res_1["mu_RQ"]),
        "res_rel_1": py(res_1["res_rel"]),
        "mu_RQ_n": py(res_n["mu_RQ"]),
        "rate_bf": py(res_n["rate_RQ"]),  # named as in `solve_point_cheb_ar`
        "res_rel_n": py(res_n["res_rel"]),
        "n_blocks_final": int(n_blocks),
        "gap": gap,
        # cost
        "m": int(m),
        "k": int(k),
        "n_apply": int(out["n_apply"]),
        "n_cycles": len(history),
        "n_cycles_loop": int(n_cycles_loop),
        "res_hist": np.array([h["res_rel"] for h in history]),
        "apply_hist": np.array([h["n_apply"] for h in history]),
        "mu_hist": np.array([h["mu"] for h in history]),
        "log": lines,
        # truncation check
        "fock_weight_a": w_a,
        "fock_weight_b": w_b,
        "tail_a": float(w_a[-k_top:].sum()),
        "tail_b": float(w_b[-k_top:].sum()),
        "k_top": int(k_top),
        "params": {kk: py(v) for kk, v in params.items()},
        "T_block": py(T_block),
        "settings": {
            "n_a": n_a, "n_b": n_b, "m": m, "k": k,
            "res_tol": res_tol, "res_tol_loop": res_tol_loop,
            "max_cycles": max_cycles, "max_cycles_loop": max_cycles_loop,
            "rtol_loop": rtol_loop, "atol_loop": atol_loop,
            "rtol_final": rtol_final, "atol_final": atol_final,
            "n_blocks_final": n_blocks_final, "n_blocks_auto": n_blocks_auto,
            "rq_resolution": rq_resolution, "seed": seed,
        },
        "timings_s": {
            "build": t_build,
            "loop": t_loop,
            "polish": t_polish,
            "final_RQ": t_final,
            "total": time.time() - t0,
        },
        "device": str(jax.devices()[0]),
    }
    if want_x_ritz:
        res["x_ritz"] = np.asarray(x_ritz)
    if want_V:
        res["V"] = V
    return res


#: Keys copied onto an `error_entry` so a failed point still says where it was.
_COORD_KEYS = ("alpha_sq", "eps_p", "kappa_b", "n_a", "n_b", "m", "k", "res_tol")


def solve_point_ks_safe(**kwargs):
    """`solve_point_ks`, returning a failure instead of raising it.

    This is what a sweep should submit as its Ray task: one unconvergeable
    point is recorded and the sweep carries on, and a raising task does not
    abort the driver's ``ray.wait`` loop and discard the points already done.
    In a warm chain a failure also costs the *next* point its starting vector,
    so the coordinator drops that chain back to a cold start after one.
    """
    try:
        return solve_point_ks(**kwargs)
    except Exception as exc:  # noqa: BLE001 - one bad point must not sink the sweep
        return error_entry(exc, **{k: kwargs.get(k) for k in _COORD_KEYS})
