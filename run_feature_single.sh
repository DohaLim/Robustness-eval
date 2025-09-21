#!/bin/bash

# Single-run Feature Evaluation Script (for quick testing)

# Positional/default arguments
# 1: Model name
# 2: Data name
# 3: Mode (evaluate|diagnose)
# 4: Level (lv1|lv2|lv3)
# 5: Shot (0|4|8|16)
# 6: Sampling method (random|best|worst) — ignored if shot==0
# 7: Seed
# 8: Engine (openai|together|vllm)
# 9: Desc method
# 10: Temperature
# 11: Num feature set
# 12: Num feature

MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
DATA=${2:-"adult"}
MODE=${3:-"diagnose"}
LEVEL=${4:-"lv1"}
SHOT=${5:-4}
METHOD=${6:-"worst"}
SEED=${7:-1}
ENGINE=${8:-"vllm"}
DESC_METHOD=${9:-"detail"}
TEMP=${10:-0.5}
NUM_FEATURE_SET=${11:-10}
NUM_FEATURE=${12:-7}

echo "[Single Run] Execution Info:"
echo "- Model: $MODEL"
echo "- Data: $DATA"
echo "- Mode: $MODE"
echo "- Level: $LEVEL"
echo "- Shot: $SHOT"
echo "- Method: $METHOD"
echo "- Seed: $SEED"
echo "- Engine: $ENGINE"
echo "- Desc Method: $DESC_METHOD"
echo "- Temp: $TEMP"
echo "- Num Feature Set: $NUM_FEATURE_SET"
echo "- Num Feature: $NUM_FEATURE"

TS=$(date +"%Y%m%d_%H%M%S") # add (optional)
if [ "$SHOT" -gt 0 ] && [ -n "$METHOD" ]; then
  OUTDIR="logs/${MODEL##*/}_${DATA}_${MODE}_${DESC_METHOD}_${LEVEL}_${SHOT}_${METHOD}_${SEED}"
else
  OUTDIR="logs/${MODEL##*/}_${DATA}_${MODE}_${DESC_METHOD}_${LEVEL}_${SHOT}_${SEED}"
fi
echo "- Output Dir: $OUTDIR"

CMD="python main.py \
  --data \"$DATA\" \
  --mode \"$MODE\" \
  --level \"$LEVEL\" \
  --shot \"$SHOT\" \
  --desc_method \"$DESC_METHOD\" \
  --seed \"$SEED\" \
  --temp \"$TEMP\" \
  --num_feature_set \"$NUM_FEATURE_SET\" \
  --num_feature \"$NUM_FEATURE\" \
  --llm-engine \"$ENGINE\" \
  --llm-name \"$MODEL\" \
  --output-dir \"$OUTDIR\""

if [ "$SHOT" -gt 0 ] && [ -n "$METHOD" ]; then
  CMD+=" \
  --sampling_method \"$METHOD\""
fi

eval $CMD

echo "[INFO] Single run finished. Results: $OUTDIR"


