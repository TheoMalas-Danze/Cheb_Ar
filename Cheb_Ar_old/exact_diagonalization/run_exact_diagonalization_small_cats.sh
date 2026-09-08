#!/bin/bash
#OAR -n exact_diag_cats
#OAR -O logs/job_%jobid%.out
#OAR -E logs/job_%jobid%.err
#OAR -l gpu=1,walltime=2:00:00
#OAR -p chuc
#OAR -q default

source $HOME/.local/bin/env
source ~/venvs/dq_gpu/bin/activate

# Print GPU model into the log, for sanity-checking the reservation
nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true

cd "$OAR_WORKDIR" || cd "$(dirname "$0")"

python exact_diagonalization_small_cats.py
