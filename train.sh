#!/bin/bash

# Usage: ./train.sh <eps1> <eps2> ... <epsN>
# Example: ./train.sh 100 200 300

PY_SCRIPT="train_wandb.py"
ENV_NAME="luchino"

XML_FILES=("snakeMotors2_14_flat.xml" "snakeMotors2_14_lowRough.xml")
EPS_ARRAY=(0.5 0.75)

for xml in "${XML_FILES[@]}"; do
  for eps in "${EPS_ARRAY[@]}"; do
    for i in {1..5}; do
      nohup conda run -n "$ENV_NAME" python "$PY_SCRIPT" \
        --xml_file "$xml" --eps "$eps" \
        > "output_${xml%.*}_eps${eps}_${i}.log" 2>&1 &
    done
  done
done
