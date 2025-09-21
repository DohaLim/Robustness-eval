#!/bin/bash

# Feature Evaluation Experiment Runner

# Positional/default arguments
# 1: Model name
# 2: Data name
# 3: Mode (evaluate|diagnose)
# 4: Engine (openai|together|vllm)
# 5: Desc method
# 6: Temperature
# 7: Num feature set
# 8: Num feature
# 9: vLLM server URL (used only when ENGINE=vllm)
# 10: swap-all flag ("true" to iterate all golden_variables, otherwise "false")

MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
DATA=${2:-"adult"}
MODE=${3:-"diagnose"}
ENGINE=${4:-"vllm"}
DESC_METHOD=${5:-"detail"}
TEMP=${6:-0.5}
NUM_FEATURE_SET=${7:-10}
NUM_FEATURE=${8:-7}
VLLM_URL=${9:-"http://localhost:8000"}
SWAP_ALL=${10:-"false"}   # "true" to automatically run for all golden_variables

# Experiment grids
LEVELS=("lv1" "lv2" "lv3")
SHOTS=(0 4 8 16)
METHODS=("random" "best" "worst")
SEEDS=(0)

echo "Execution Info:"
echo "- Model: $MODEL"
echo "- Data: $DATA"
echo "- Mode: $MODE"
echo "- Engine: $ENGINE"
echo "- Desc Method: $DESC_METHOD"
echo "- Temp: $TEMP"
echo "- Num Feature Set: $NUM_FEATURE_SET"
echo "- Num Feature: $NUM_FEATURE"
[ "$ENGINE" = "vllm" ] && echo "- vLLM URL: $VLLM_URL"
echo "- Swap-All: $SWAP_ALL"

get_golden_vars() {
python - <<'PY' "$DATA"
import sys
data = sys.argv[1]
import data_utils
setup = data_utils.setup(data)
vars_ = setup.get("golden_variables") or []
for v in vars_:
    print(v)
PY
}

GOLDEN_VARS=()
if [ "$SWAP_ALL" = "true" ]; then
  GOLDEN_VARS=($(get_golden_vars))
fi

for _LEVEL in "${LEVELS[@]}"; do
  for _SHOT in "${SHOTS[@]}"; do
    # Choose methods only when shot > 0; otherwise, use a placeholder to skip passing sampling_method
    if [ "$_SHOT" -gt 0 ]; then
      _METHOD_LIST=("${METHODS[@]}")
    else
      _METHOD_LIST=("")
    fi

    for _METHOD in "${_METHOD_LIST[@]}"; do
      for _SEED in "${SEEDS[@]}"; do
        SWAP_LIST=( "__NONE__" )
        if [ "$_SHOT" -gt 0 ] && [ "$_METHOD" = "random" ] && [ "$SWAP_ALL" = "true" ]; then
          SWAP_LIST=( "__NONE__" "${GOLDEN_VARS[@]}" )
        fi

        for _SWAP in "${SWAP_LIST[@]}"; do
          TS=$(date +"%Y%m%d_%H%M%S")
          if [ "$_SWAP" = "__NONE__" ]; then
            SWAP_TAG="noswap"
          else
            SWAP_TAG="${_SWAP//\//_}"
          fi

          if [ "$_SHOT" -gt 0 ] && [ -n "$_METHOD" ]; then
            OUTDIR="logs/${MODEL##*/}_${DATA}_${MODE}_${_LEVEL}_${DESC_METHOD}_s${_SHOT}_${_METHOD}_${SWAP_TAG}_seed${_SEED}"
          else
            OUTDIR="logs/${MODEL##*/}_${DATA}_${MODE}_${_LEVEL}_${DESC_METHOD}_s${_SHOT}_${SWAP_TAG}_seed${_SEED}"
          fi

          echo -e "\nRunning: level=${_LEVEL}, shot=${_SHOT}, method=${_METHOD:-none}, seed=${_SEED}, swap=${_SWAP/__NONE__/none}"
          echo "- Output Dir: $OUTDIR"
          
          CMD="python main.py \
            --data \"$DATA\" \
            --mode \"$MODE\" \
            --level \"$_LEVEL\" \
            --shot \"$_SHOT\" \
            --desc_method \"$DESC_METHOD\" \
            --seed \"$_SEED\" \
            --temp \"$TEMP\" \
            --num_feature_set \"$NUM_FEATURE_SET\" \
            --num_feature \"$NUM_FEATURE\" \
            --llm-engine \"$ENGINE\" \
            --llm-name \"$MODEL\" \
            --output-dir \"$OUTDIR\""

          # sampling_method (only when available)
          if [ -n "$_METHOD" ]; then
            CMD+=" \
            --sampling_method \"$_METHOD\""
          fi
          # swap (only when available)
          if [ "$_SWAP" != "__NONE__" ]; then
            CMD+=" \
            --swap \"$_SWAP\""
          fi
          # vLLM URL (only when engine is vllm)
          if [ "$ENGINE" = "vllm" ]; then
            CMD+=" \
            --vllm-server-url \"$VLLM_URL\""
          fi

          # Execute
          eval $CMD

          echo "[INFO] Completed: level=${_LEVEL}, shot=${_SHOT}, method=${_METHOD:-none}, seed=${_SEED}, swap=${_SWAP/__NONE__/none}"
        done
      done
    done
  done
done

echo -e "\nAll experiments finished."


