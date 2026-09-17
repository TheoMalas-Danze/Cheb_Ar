"""Submit a sweep driver to the anb-compute Ray cluster, from anb-dev.

Replaces the OAR `.sh` job files. There is no queue, no walltime and no worker
count here: the driver submits one task per sweep point and the autoscaler sizes
the gpu pool for whatever is pending.

    pixi run python scripts/cluster/submit.py sweep_alpha -- --n-a 20 --n-b 8

Everything after ``--`` is forwarded verbatim to the driver, except ``--output``
which this script fills in with a fresh, dated run prefix (pass ``--output`` to
override). Run prefixes are never reused: ``save_output`` overwrites silently.

Prerequisites, in the shell that submits:

    export VAULT_ADDR=https://vault.int.alice-bob.com
    vault login -method=oidc role=research_core_user
    export RAY_AUTH_MODE=token
    export RAY_AUTH_TOKEN=$(vault kv get -field=token kv-it/prod/ray/auth-token)
"""

import argparse
import datetime as dt
import os
import sys
import uuid

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(REPO, "scripts")

# Shipped by PATH, not by name, so local edits actually reach the workers.
CHEB_AR = os.path.join(REPO, "src", "cheb_ar")
# Upstream dynamiqs v0.3.6 + the `mesolve_fast` patch; the cluster image carries
# a stock 0.3.6, which lacks it. Drop this once the patch is upstreamed.
# Derived, not hardcoded: a `dynamiqs` checkout beside this repo, or $DYNAMIQS_SRC.
DYNAMIQS = os.environ.get("DYNAMIQS_SRC") or os.path.join(
    os.path.dirname(REPO), "dynamiqs", "dynamiqs"
)

BUCKET = "gs://anb-ray-results/cheb-ar"
DRIVERS = ("sweep_alpha", "sweep_eps_p", "exact_diagonalization")


def main():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("driver", choices=DRIVERS, help="which sweep to run")
    p.add_argument("--output", default=None,
                   help="GCS prefix for this run (default: a fresh dated one)")
    p.add_argument("--wait", action="store_true",
                   help="block until the job ends and print its logs "
                        "(default: return the job id and exit)")
    p.add_argument("driver_args", nargs=argparse.REMAINDER,
                   help="passed through to the driver, after --")
    args = p.parse_args()

    if os.environ.get("RAY_AUTH_MODE") != "token":
        sys.exit("RAY_AUTH_MODE is not set to 'token' — see the header of this file.")
    for path in (CHEB_AR, DYNAMIQS):
        if not os.path.isdir(path):
            sys.exit(f"not a directory: {path}")

    from anb_compute import ray as acr

    extra = args.driver_args
    if extra and extra[0] == "--":
        extra = extra[1:]

    uri = args.output or (
        f"{BUCKET}/{args.driver}/"
        f"{dt.date.today():%Y-%m-%d}-{uuid.uuid4().hex[:8]}/"
    )

    entrypoint = " ".join(
        [f"python {args.driver}.py", "--output", uri, *extra]
    )
    print(f"entrypoint: {entrypoint}")
    print(f"results   : {uri}results.json")

    runtime_env = acr.build_runtime_env(
        py_modules=[CHEB_AR, DYNAMIQS],
        # The driver script itself travels as the working dir; its *contents*
        # land at the driver's CWD, hence `python sweep_alpha.py`, not
        # `python scripts/sweep_alpha.py`.
        base={"working_dir": SCRIPTS},
    )

    out = acr.submit_job(
        entrypoint,
        runtime_env=runtime_env,
        # Put the driver on a worker rather than the shared 4-cpu head: it holds
        # every point's result before writing them out.
        entrypoint_num_cpus=1,
        wait=args.wait,
    )
    if args.wait:
        print(f"job finished: {out}")
    else:
        print(f"job id: {out}\n"
              f"  follow : ray job logs --follow {out} "
              f"--address https://compute.int.alice-bob.com\n"
              f"  stop   : python -c \"from anb_compute import ray as acr; "
              f"acr.stop_job('{out}')\"")
    print(
        "\nread the results back with:\n"
        "  from anb_compute import ray as acr\n"
        "  import json\n"
        f"  acr.load_output('{uri}results.json', mode='r', loader=json.load,\n"
        "                  token=acr.gcs_credentials())"
    )


if __name__ == "__main__":
    main()
