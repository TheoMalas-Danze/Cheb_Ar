#!/bin/bash
#OAR -n cheb_ar_vs_alpha_vs_eps
#OAR -O logs/job_%jobid%.out
#OAR -E logs/job_%jobid%.err
#OAR -l gpu=1,walltime=0:10:00
#OAR -p chuc
#OAR -q default

source $HOME/.local/bin/env
source ~/venvs/dq_gpu/bin/activate

# Print GPU model into the log, for sanity-checking the reservation
nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true

cd "$OAR_WORKDIR" || cd "$(dirname "$0")"

python cheb_ar_loop_alpha_eps.py
