#!/usr/bin/env bash
set -u
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=2
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

SCRIPT="./r2bert_hybrid_cls_topk_cv.py"
ASAP="./asap-aes/training_set_rel3.tsv"

OUTROOT="./out_r2bert_hybrid_topk_sweep"
TIMES_CSV="${OUTROOT}/times.csv"
FINAL_CSV="${OUTROOT}/topk_sweep_results_with_prompts.csv"
PLOT_PNG="${OUTROOT}/topk_vs_avg_qwk.png"

mkdir -p "${OUTROOT}"
echo "k,seconds,status,out_dir" > "${TIMES_CSV}"

K_LIST=(1 3 5 10 20 50 768)
TOTAL=${#K_LIST[@]}

fmt_time () {
  local T=$1
  local h=$((T/3600)); local m=$(((T%3600)/60)); local s=$((T%60))
  printf "%02d:%02d:%02d" "$h" "$m" "$s"
}

avg_success_seconds () {
  awk -F, 'NR>1 && $3==0 {sum+=$2; n++} END{ if(n>0) printf "%.0f", sum/n; else print 0 }' "${TIMES_CSV}"
}

for idx in "${!K_LIST[@]}"; do
  K="${K_LIST[$idx]}"
  i=$((idx+1))
  OUTDIR="${OUTROOT}/k${K}"
  mkdir -p "${OUTDIR}"

  echo "==================== [${i}/${TOTAL}] Running HYBRID K=${K} ===================="

  start_ts=$(date +%s)

  stdbuf -oL -eL python -u "${SCRIPT}" \
    --asap_path "${ASAP}" \
    --prompt_id 0 \
    --cv_folds 5 \
    --num_epochs 60 \
    --top_k_pool "${K}" \
    --batch_size 32 \
    --lr 2e-5 \
    --tau1 1e-4 \
    --weight_decay 0.01 \
    --dropout 0.1 \
    --out_dir "${OUTDIR}" \
    --amp \
    2>&1 | tee "${OUTDIR}/run.log"

  status=${PIPESTATUS[0]}

  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))

  echo "${K},${elapsed},${status},${OUTDIR}" >> "${TIMES_CSV}"

  if [[ $status -ne 0 ]]; then
    echo "K=${K} FAILED (exit=${status}). Log: ${OUTDIR}/run.log"
  else
    echo "K=${K} DONE in $(fmt_time ${elapsed})"
  fi

  avg=$(avg_success_seconds)
  if [[ "$avg" -gt 0 ]]; then
    remaining=$((TOTAL - i))
    eta=$((remaining * avg))
    echo "[Sweep ETA] ~$(fmt_time ${eta}) remaining (avg per successful K ≈ $(fmt_time ${avg}))"
  fi
done

python ./summarize_topk_sweep.py \
  --times_csv "${TIMES_CSV}" \
  --out_csv "${FINAL_CSV}" \
  --plot_png "${PLOT_PNG}"

echo "========================================================"
echo "Saved times CSV:   ${TIMES_CSV}"
echo "Saved final CSV:   ${FINAL_CSV}"
echo "Saved plot PNG:    ${PLOT_PNG}"