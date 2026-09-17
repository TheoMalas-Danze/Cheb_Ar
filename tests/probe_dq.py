"""Does the image's dynamiqs have the fork's `mesolve_fast`?

The prod-gpu image carries dynamiqs 0.3.6, but Cheb_Ar pins a *fork*
(TheoMalas-Danze/dynamiqs @ b52cb66) whose single extra commit adds
`mesolve_fast`. `ChebAr._build_propagator` calls it whenever `jump_ops_LdL`
is passed, which is the whole interaction-frame path. So the question is
whether the image's stock dynamiqs is enough, or the fork must be shipped.

    pixi run python probe_dq.py
"""

import os
import sys


def dq_probe():
    import importlib.metadata as md

    import dynamiqs as dq

    out = {
        "dynamiqs": md.version("dynamiqs"),
        "has_mesolve_fast": hasattr(dq, "mesolve_fast"),
        "has_mesolve": hasattr(dq, "mesolve"),
        "has_mepropagator": hasattr(dq, "mepropagator"),
        "has_set_precision": hasattr(dq, "set_precision"),
        "has_vectorize": hasattr(dq, "vectorize"),
        "has_plot_wigner": hasattr(getattr(dq, "plot", None), "wigner"),
    }
    # Options(assume_hermitian=...) is used on every propagator call.
    try:
        dq.Options(assume_hermitian=False)
        out["options_assume_hermitian"] = True
    except Exception as exc:
        out["options_assume_hermitian"] = f"{type(exc).__name__}: {exc}"
    return out


if __name__ == "__main__":
    if os.environ.get("RAY_AUTH_MODE") != "token":
        sys.exit("set RAY_AUTH_MODE=token and RAY_AUTH_TOKEN first")

    from anb_compute import ray as acr

    print("stock dynamiqs on the gpu image:")
    for k, v in acr.run(dq_probe, num_gpus=1, num_cpus=8, poll_seconds=2).items():
        print(f"  {k}: {v}")
