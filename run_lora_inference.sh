#!/bin/bash

# # vehicle_00015, r8, lr0.0001
# python my_scripts/P10_run_lora_inference.py \
#   --experiment_name "r8_iter3000_bs8_lr0.0001_seed0" \
#   --prompt "The person is driving a purple car on an empty road with blue sky in the background." \
#   --inference_name "vehicle_00015" \
#   --num_videos 3 \
#   --num_steps 60 \
#   --fps 24 \
#   --guidance 6.5

# # vehicle_00015, r16, lr5e-05
# python my_scripts/P10_run_lora_inference.py \
#   --experiment_name "r16_iter3000_bs8_lr5e-05_seed0" \
#   --prompt "The person is driving a purple car on an empty road with blue sky in the background." \
#   --inference_name "vehicle_00015" \
#   --num_videos 3 \
#   --num_steps 60 \
#   --fps 24 \
#   --guidance 6.5

# # vehicle_00015, r16, lr0.0001
# python my_scripts/P10_run_lora_inference.py \
#   --experiment_name "r16_iter3000_bs8_lr0.0001_seed0" \
#   --prompt "The person is driving a purple car on an empty road with blue sky in the background." \
#   --inference_name "vehicle_00015" \
#   --num_videos 3 \
#   --num_steps 60 \
#   --fps 24 \
#   --guidance 6.5



# vehicle_00025, r8, lr0.0001
python my_scripts/P10_run_lora_inference.py \
  --experiment_name "r8_iter3000_bs8_lr0.0001_seed0" \
  --prompt "A truck on the side of the road next to a house." \
  --inference_name "vehicle_00025" \
  --num_videos 3 \
  --num_steps 60 \
  --fps 24 \
  --guidance 6.5

# vehicle_00025, r16, lr5e-05
python my_scripts/P10_run_lora_inference.py \
  --experiment_name "r16_iter3000_bs8_lr5e-05_seed0" \
  --prompt "A truck on the side of the road next to a house." \
  --inference_name "vehicle_00025" \
  --num_videos 3 \
  --num_steps 60 \
  --fps 24 \
  --guidance 6.5

# vehicle_00025, r16, lr0.0001
python my_scripts/P10_run_lora_inference.py \
  --experiment_name "r16_iter3000_bs8_lr0.0001_seed0" \
  --prompt "A truck on the side of the road next to a house." \
  --inference_name "vehicle_00025" \
  --num_videos 3 \
  --num_steps 60 \
  --fps 24 \
  --guidance 6.5



# vehicle_00048, r8, lr0.0001
python my_scripts/P10_run_lora_inference.py \
  --experiment_name "r8_iter3000_bs8_lr0.0001_seed0" \
  --prompt "A person is driving a car on a city street, and there are several cars parked on the side of the road." \
  --inference_name "vehicle_00048" \
  --num_videos 3 \
  --num_steps 60 \
  --fps 24 \
  --guidance 6.5

# vehicle_00048, r16, lr5e-05
python my_scripts/P10_run_lora_inference.py \
  --experiment_name "r16_iter3000_bs8_lr5e-05_seed0" \
  --prompt "A person is driving a car on a city street, and there are several cars parked on the side of the road." \
  --inference_name "vehicle_00048" \
  --num_videos 3 \
  --num_steps 60 \
  --fps 24 \
  --guidance 6.5

# vehicle_00048, r16, lr0.0001
python my_scripts/P10_run_lora_inference.py \
  --experiment_name "r16_iter3000_bs8_lr0.0001_seed0" \
  --prompt "A person is driving a car on a city street, and there are several cars parked on the side of the road." \
  --inference_name "vehicle_00048" \
  --num_videos 3 \
  --num_steps 60 \
  --fps 24 \
  --guidance 6.5
