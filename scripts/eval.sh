#!/bin/bash
set -eo pipefail

BUDGET="${1:?usage: bash scripts/eval.sh <budget: 192|128|64|576> [general|grounding|all|task,...] [limit]}"
GROUP="${2:-general}"
LIMIT="${3:-}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the llava-v1.5-7b checkpoint}"
NPROC="${NPROC:-$(nvidia-smi -L | wc -l)}"
OUTPUT="${OUTPUT:-./results/budget${BUDGET}}"

GENERAL="vqav2_testdev,gqa,mmbench_en_dev,ok_vqa_val2014,scienceqa_img,ai2d,textvqa_val,ocrbench,docvqa_val,chartqa,pope"
GROUNDING="refcoco_bbox_rec_val,refcoco_bbox_rec_testA,refcoco_bbox_rec_testB,refcoco+_bbox_rec_val,refcoco+_bbox_rec_testA,refcoco+_bbox_rec_testB,refcocog_bbox_rec_val,refcocog_bbox_rec_test"

case "${GROUP}" in
    general) TASKS="${GENERAL}" ;;
    grounding) TASKS="${GROUNDING}" ;;
    all) TASKS="${GENERAL},${GROUNDING}" ;;
    *) TASKS="${GROUP}" ;;
esac

ARGS=()
[ -n "${LIMIT}" ] && ARGS+=(--limit "${LIMIT}")

LMMS_EVAL_PLUGINS=bifocal python -m accelerate.commands.launch --num_processes="${NPROC}" -m lmms_eval \
    --model llava_bifocal \
    --model_args "pretrained=${MODEL_PATH},budget=${BUDGET}" \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "budget${BUDGET}" \
    --output_path "${OUTPUT}" \
    "${ARGS[@]}"
