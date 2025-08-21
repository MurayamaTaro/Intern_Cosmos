#!bin/bash

echo "Waiting for 3 hours..."
sleep $((3*3600))

echo "Running P11_run_lora_inference_sweep.py..."
python my_scripts/P11_run_lora_inference_sweep.py

echo "Running P09_run_hyperparameter_sweep.py..."
python my_scripts/P09_run_hyperparameter_sweep.py

