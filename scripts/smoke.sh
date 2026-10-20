#!/bin/bash
set -eo pipefail

cd "$(dirname "$0")/.."
bash install.sh check
for BUDGET in 576 128; do
    NPROC=1 OUTPUT="./results/smoke/budget${BUDGET}" bash scripts/eval.sh "${BUDGET}" refcoco_bbox_rec_testA 8
done
