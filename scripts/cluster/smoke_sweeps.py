"""Smoke-test the refactored sweep machinery on the cluster, cheaply.

Exercises the code paths the sweep drivers and notebooks rely on, at a small
Fock truncation so it costs seconds rather than GPU-hours:

1. fan-out of ``solve_point_safe`` over independent points (the alpha_sq shape);
2. the sequential warm-start chain with the eigenbasis rotation (the eps_p
   shape) -- including that ``want_V`` really comes back and ``warm_start``
   flips to True after the first point;
3. the failure path: a bad point is *returned* as an error entry, not raised,
   so it cannot sink the sweep.

It runs the ChebAr pipeline (``solve_point_cheb_ar_safe``), whose small
settings are known to be quick. The Krylov-Schur pipeline shares the same
contract (``x0`` / ``V_prev`` / ``want_V`` / error entries) but has no
smoke settings calibrated yet.

    pixi run python scripts/cluster/smoke_sweeps.py
"""

import os
import pathlib
import sys

# Derived, never hardcoded: this file is scripts/cluster/<name>.py, so the repo root
# is three levels up. (CLAUDE.md: absolute paths were stripped from this repo once
# already.) The patched dynamiqs checkout is expected beside the repo; override
# with $DYNAMIQS_SRC.
REPO = pathlib.Path(__file__).resolve().parents[2]
PKG = str(REPO / "src" / "floquet_lindblad")
DYNAMIQS = os.environ.get("DYNAMIQS_SRC") or str(REPO.parent / "dynamiqs" / "dynamiqs")

# Small enough to be quick, big enough that the ellipse fit has something to do.
# Deliberately no alpha_sq/eps_p here: those are the swept coordinates, passed
# explicitly at each call site. Putting one in both places is a
# "got multiple values for keyword argument" TypeError.
SMALL = dict(n_a=15, n_b=6, cheb_degree=6,
             m_arnoldi_0=40, m_arnoldi=60, margin=1e-2)
CHAIN_ALPHA_SQ = 2.0


def driver(settings):
    """Runs on a worker; fans out and chains, exactly as the real sweeps do."""
    import numpy as np
    import ray

    from anb_compute import ray as acr
    from floquet_lindblad.pipeline import solve_point_cheb_ar_safe as solve_point_safe

    report = {}

    # --- 1. independent fan-out (alpha_sq shape) -------------------------
    alphas = [1.5, 2.0, 2.5]
    refs = {
        acr.submit(solve_point_safe, alpha_sq=a, eps_p=0.1, want_x_ritz=False,
                   num_gpus=1, num_cpus=8, **settings): a
        for a in alphas
    }
    fan = []
    pending = list(refs)
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        r = ray.get(ready[0])
        fan.append((refs[ready[0]], r.get("error") or complex(r["rate_bf"]).real))
    report["fanout"] = sorted(fan, key=lambda t: t[0])

    # --- 2. sequential warm-start chain (eps_p shape) --------------------
    chain = []
    prev = None
    for eps_p in (0.1, 0.3, 0.5):
        warm = prev is not None
        r = ray.get(acr.submit(
            solve_point_safe,
            alpha_sq=2.0,
            eps_p=eps_p,
            kappa_b=float(np.sin(eps_p) / np.sin(0.1) * (5 / 10.4)),
            x0=prev["x_ritz"] if warm else None,
            V_prev=prev["V"] if warm else None,
            want_x_ritz=True, want_V=True,
            num_gpus=1, num_cpus=8, **settings,
        ))
        if "error" in r:
            chain.append((eps_p, r["error"]))
            prev = None
            continue
        chain.append((eps_p, complex(r["rate_bf"]).real, r["warm_start"],
                      None if r.get("V") is None else np.shape(r["V"])))
        prev = r
    report["chain"] = chain

    # --- 3. failure path: must return, not raise -------------------------
    bad = ray.get(acr.submit(
        solve_point_safe, alpha_sq=2.0, eps_p=0.1,
        n_a=0, n_b=0,  # degenerate: blows up inside the builder
        cheb_degree=6, m_arnoldi_0=8, m_arnoldi=8, margin=1e-2,
        num_gpus=1, num_cpus=8,
    ))
    report["failure_returns_entry"] = ("error" in bad, bad.get("error", "")[:110])
    report["failure_keeps_coords"] = {k: bad.get(k) for k in ("alpha_sq", "eps_p", "n_a")}
    return report


if __name__ == "__main__":
    if os.environ.get("RAY_AUTH_MODE") != "token":
        sys.exit("set RAY_AUTH_MODE=token and RAY_AUTH_TOKEN first")

    from anb_compute import ray as acr

    out = acr.run(
        driver, SMALL,
        runtime_env=acr.build_runtime_env(py_modules=[PKG, DYNAMIQS]),
        poll_seconds=5,
    )
    print("\n=== fan-out (independent points) ===")
    for row in out["fanout"]:
        print("  alpha_sq=%-5s -> %s" % row)
    print("\n=== warm-start chain (eps_p, basis rotated each step) ===")
    for row in out["chain"]:
        print("  eps_p=%-5s -> rate=%-12.6g warm=%-5s V.shape=%s" % row
              if len(row) == 4 else "  eps_p=%-5s -> %s" % row)
    print("\n=== failure path ===")
    print("  returned an error entry:", out["failure_returns_entry"][0])
    print("  message:", out["failure_returns_entry"][1])
    print("  coords preserved:", out["failure_keeps_coords"])
