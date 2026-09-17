"""Verify the rebased dynamiqs (v0.3.6 + mesolve_fast) works on a GPU worker.

Ships the local dynamiqs checkout via py_modules, which prepends to the worker's
sys.path and so shadows the image's stock 0.3.6. Then does more than import it:
runs an actual `mesolve_fast` and checks it agrees with plain `mesolve`, which is
the property Cheb_Ar relies on.

    pixi run python probe_fork.py
"""

import os
import pathlib
import sys

# Derived, never hardcoded: this file is tests/<name>.py, so the repo root is
# two levels up. (CLAUDE.md: absolute paths were stripped from this repo once
# already.) The patched dynamiqs checkout is expected beside the repo; override
# with $DYNAMIQS_SRC.
REPO = pathlib.Path(__file__).resolve().parents[1]
CHEB_AR = str(REPO / "src" / "cheb_ar")
DYNAMIQS = os.environ.get("DYNAMIQS_SRC") or str(REPO.parent / "dynamiqs" / "dynamiqs")

FORK = DYNAMIQS


def fork_probe():
    import importlib.metadata as md

    import dynamiqs as dq
    import jax.numpy as jnp
    import numpy as np

    out = {
        "dynamiqs_file": dq.__file__,
        "dynamiqs_version": dq.__version__,
        "dist_version": md.version("dynamiqs"),
        "has_mesolve_fast": hasattr(dq, "mesolve_fast"),
        "backend": __import__("jax").default_backend(),
    }
    if not out["has_mesolve_fast"]:
        return out

    # A small damped two-level + cavity problem, in double precision, on the GPU.
    dq.set_precision("double")
    n = 8
    a = dq.destroy(n)
    H = 1.0 * dq.dag(a) @ a
    kappa = 0.3
    L = np.sqrt(kappa) * a
    jump_ops = [L]
    LdL = [dq.dag(L) @ L]
    rho0 = dq.coherent(n, 1.0) @ dq.dag(dq.coherent(n, 1.0))
    tsave = jnp.array([0.0, 0.7])
    opts = dq.Options(assume_hermitian=False)
    method = dq.method.Tsit5(rtol=1e-9, atol=1e-10)

    # v0.3.6 flattened the public mesolve's options into keyword arguments
    # (upstream #1088), so `options=` is gone here...
    slow = dq.mesolve(H, jump_ops, rho0, tsave, method=method, assume_hermitian=False)
    # ...and mesolve_fast now mirrors that flattened style.
    fast = dq.mesolve_fast(
        H, jump_ops, LdL, rho0, tsave, method=method, assume_hermitian=False
    )
    # The old style must now be rejected, not silently ignored.
    try:
        dq.mesolve_fast(H, jump_ops, LdL, rho0, tsave, options=opts)
        out["rejects_old_options_kwarg"] = False
    except TypeError:
        out["rejects_old_options_kwarg"] = True

    rs = slow.states[-1].to_jax()
    rf = fast.states[-1].to_jax()
    out["state_dtype"] = str(rf.dtype)
    out["max_abs_diff_vs_mesolve"] = float(jnp.max(jnp.abs(rs - rf)))
    out["agrees"] = bool(jnp.allclose(rs, rf, atol=1e-8))
    # Photon number should decay as exp(-kappa t) from |alpha|^2 = 1.
    out["n_final"] = float(jnp.real(jnp.trace(rf @ (dq.dag(a) @ a).to_jax())))
    out["n_expected"] = float(np.exp(-kappa * 0.7))
    return out


if __name__ == "__main__":
    if os.environ.get("RAY_AUTH_MODE") != "token":
        sys.exit("set RAY_AUTH_MODE=token and RAY_AUTH_TOKEN first")
    if not os.path.isdir(FORK):
        sys.exit(f"no such package dir: {FORK}")

    from anb_compute import ray as acr

    print(f"shipping {FORK} to a gpu worker ...")
    result = acr.run(
        fork_probe,
        num_gpus=1,
        num_cpus=8,
        poll_seconds=2,
        runtime_env=acr.build_runtime_env(py_modules=[FORK]),
    )
    for k, v in result.items():
        print(f"  {k}: {v}")
