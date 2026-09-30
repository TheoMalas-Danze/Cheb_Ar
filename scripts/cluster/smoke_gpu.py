"""Smoke test for the anb-compute Ray cluster, from the anb-dev container.

Run with:  pixi run python scripts/cluster/smoke_gpu.py

Prerequisites (see the anb-compute "Ray setup" user guide, laptop path):

    export VAULT_ADDR=https://vault.int.alice-bob.com
    vault login -method=oidc role=research_core_user
    export RAY_AUTH_MODE=token
    export RAY_AUTH_TOKEN=$(vault kv get -field=token kv-it/prod/ray/auth-token)

Both variables must be set in the shell that submits: Ray silently attaches no
auth header without RAY_AUTH_MODE=token, which is the usual cause of a 401.
The Vault session has an 8h TTL, so re-run `vault login` when it expires.
"""

import os
import sys


def check_prerequisites() -> None:
    """Fail early and loudly, instead of hanging or returning an opaque 401."""
    import urllib.error
    import urllib.request

    # 1. Network: 401 is the *healthy* answer here (the cluster refusing an
    #    unauthenticated request). A timeout means routing, not auth: on
    #    WireGuard, add 10.5.0.0/16 and 10.6.0.0/16 to AllowedIPs.
    url = "https://compute.int.alice-bob.com/api/version"
    try:
        code = urllib.request.urlopen(url, timeout=15).status
    except urllib.error.HTTPError as exc:
        code = exc.code
    except OSError as exc:
        sys.exit(f"[prereq 1] cannot reach {url}: {exc}\n  -> not on the VPN, or routing is missing")
    print(f"[prereq 1] {url} -> {code} (401 or 200 both mean reachable)")

    # 2. Auth.
    missing = [v for v in ("RAY_AUTH_MODE", "RAY_AUTH_TOKEN") if not os.environ.get(v)]
    if missing:
        sys.exit(f"[prereq 2] unset: {', '.join(missing)}\n  -> see the docstring at the top of this file")
    if os.environ["RAY_AUTH_MODE"] != "token":
        sys.exit(f"[prereq 2] RAY_AUTH_MODE={os.environ['RAY_AUTH_MODE']!r}, expected 'token'")
    print("[prereq 2] RAY_AUTH_MODE=token, RAY_AUTH_TOKEN set")

    # 3. Versions: only the client JobSubmissionClient has to match the cluster.
    import ray

    print(f"[prereq 3] python {sys.version.split()[0]} (want 3.13.x), ray {ray.__version__} (want 2.56.0)")


def hello(x):
    """Plain CPU task: proves auth, submission and result read-back."""
    import socket

    return f"{x * 2} computed on {socket.gethostname()}"


def gpu_probe():
    """Did I really get a GPU?

    The user guide's own snippet for this shells out to ``nvidia-smi -L``, but
    that binary is not on PATH in the worker image, so it raises
    ``FileNotFoundError`` and tells you nothing. Ask the runtime instead.
    """
    import glob
    import os
    import shutil
    import subprocess

    import ray

    out = {
        # What Ray thinks it gave us, and what it exported for CUDA.
        "ray_gpu_ids": ray.get_gpu_ids(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        # Does the node actually carry a driver and devices?
        "dev_nvidia": sorted(glob.glob("/dev/nvidia*")),
        "driver_version": None,
        "nvidia_smi": shutil.which("nvidia-smi"),
        # Which env the task runs in (prod vs prod-gpu).
        "sys_prefix": __import__("sys").prefix,
    }
    try:
        with open("/proc/driver/nvidia/version") as f:
            out["driver_version"] = f.read().strip()
    except OSError as exc:
        out["driver_version"] = f"unreadable: {exc}"
    if out["nvidia_smi"]:
        out["nvidia_smi_L"] = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True
        ).stdout.strip()
    return out


def jax_probe():
    """Does the *image's* jax see that GPU?

    anb-core-dependencies[ml] pins a plain ``jax==0.10.2``, with no
    ``jax-cuda12-plugin`` / ``jax-cuda12-pjrt`` among its declared
    dependencies. If the prod image does not bake those in separately, jax
    falls back to CPU on a gpu node and Cheb_Ar's ``require_gpu`` guard trips.
    This probe is what decides whether Cheb_Ar needs a runtime_env ``pip``
    entry for ``jax[cuda12]`` or an image rebuild.
    """
    import importlib.metadata as md
    import sys

    import jax

    installed = sorted(
        d.metadata["Name"]
        for d in md.distributions()
        if (d.metadata["Name"] or "").lower().startswith(("jax", "nvidia-cuda"))
    )
    # Cheb_Ar is double precision throughout, so x64 must survive on the worker.
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    out = {
        "sys_prefix": sys.prefix,
        "jax": jax.__version__,
        "default_backend": jax.default_backend(),
        "devices": [f"{d.platform}:{d.id}" for d in jax.devices()],
        "jax_related_dists": installed,
        "x64_dtype": str(jnp.zeros(1).dtype),
    }
    # A real matmul on the default device: proves the GPU actually computes,
    # not just that a device object exists.
    x = jnp.ones((256, 256), dtype=jnp.complex128)
    y = (x @ x).block_until_ready()
    out["matmul_device"] = str(next(iter(y.devices())))
    out["matmul_ok"] = bool(y[0, 0] == 256 + 0j)

    # Does the worker already have what Cheb_Ar needs?
    for pkg in ("diffrax", "dynamiqs", "qutip", "matplotlib", "scipy"):
        try:
            out[f"has_{pkg}"] = md.version(pkg)
        except md.PackageNotFoundError:
            out[f"has_{pkg}"] = None
    return out


if __name__ == "__main__":
    check_prerequisites()

    # Imported after the RAY_AUTH_MODE check: Ray reads that variable once, at its
    # first import, and caches it.
    from anb_compute import ray as acr

    # Each step is independent: a failure in one must not skip the others, since
    # the jax probe is the one that actually decides the Cheb_Ar port.
    steps = [
        # poll_seconds=2 instead of the default 10, so a trivial job returns promptly.
        ("hello on the cpu pool", hello, (21,), {"poll_seconds": 2}),
        # num_gpus > 0 is what routes the task to the gpu pool (on-demand only).
        # It also makes it run as a task on a worker, not inline in the driver.
        ("gpu_probe on the gpu pool", gpu_probe, (), {"num_gpus": 1, "num_cpus": 8, "poll_seconds": 2}),
        ("jax_probe on the gpu pool", jax_probe, (), {"num_gpus": 1, "num_cpus": 8, "poll_seconds": 2}),
    ]

    failures = 0
    for i, (label, fn, args, kwargs) in enumerate(steps, start=1):
        print(f"\n[{i}/{len(steps)}] {label} ...")
        try:
            result = acr.run(fn, *args, **kwargs)
        except Exception as exc:
            failures += 1
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue
        if isinstance(result, dict):
            for k, v in result.items():
                print(f"  {k}: {v}")
        else:
            print("  ->", result)

    sys.exit(1 if failures else 0)
