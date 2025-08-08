#!/bin/bash

# Usage: ./train.sh <eps1> <eps2> ... <epsN>
# Example: ./train.sh 100 200 300

PY_SCRIPT="train_wandb.py"
ENV_NAME="snakebot3"

XML_FILES=("snakeMotors2_14_highRough.xml" "snakeMotors2_14_medRough.xml" )
EPS_ARRAY=(0.25 0.5 0.75 1)
find . -maxdepth 1 -type f -name "*.out" -delete
find . -maxdepth 1 -type f -name "*.log" -delete
for xml in "${XML_FILES[@]}"; do
  for eps in "${EPS_ARRAY[@]}"; do
      nohup conda run -n "$ENV_NAME" python3.11 "$PY_SCRIPT" \
        --xml_file "$xml" --eps "$eps" \
        > "output_${xml%.*}_eps${eps}_.log"  &
  done
done
