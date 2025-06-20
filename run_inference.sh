#!/bin/bash

# T2I 車0
python3 inference/batch_inference.py \
--model_type text2world \
--prompt "A sports car accelerates smoothly along a coastal highway at sunset. The camera follows behind, capturing the reflections of the sun on the glossy curved body. The suspension gently compresses as the car takes a slight curve, tires gripping the asphalt with subtle deformation. Waves crash against the nearby rocks as seagulls fly overhead." \
--num_seeds 10 \
--num_steps 70 \
--fps 24 \
--output_dir car_0

# T2I 車1
python3 inference/batch_inference.py \
--model_type text2world \
--prompt "Multiple cars move through a busy intersection in a futuristic city. Traffic lights change, pedestrians cross, and autonomous vehicles navigate smoothly around each other. Reflections from skyscrapers glide over the car surfaces, while drones fly overhead. The camera follows one car from above, then smoothly transitions to street-level perspective." \
--num_seeds 10 \
--num_steps 70 \
--fps 24 \
--output_dir car_1

# T2I 猫0
python3 inference/batch_inference.py \
--model_type text2world \
--prompt "A fluffy cat lies on a soft blanket, slowly kneading with its front paws. Its fur deforms realistically with each paw press. The camera focuses on the paws, showing individual toe movements and subtle fur shifts. The cat blinks slowly, purring softly, while sunlight casts warm reflections on its fur." \
--num_seeds 10 \
--num_steps 70 \
--fps 24 \
--output_dir cat_0

# T2I 猫1
python3 inference/batch_inference.py \
--model_type text2world \
--prompt "Three kittens chase each other playfully in a living room. They dart between furniture, jump over cushions, and slide slightly on the polished floor. Their fur shifts dynamically with every movement. The camera smoothly tracks their energetic chase while keeping focus on one kitten at a time." \
--num_seeds 10 \
--num_steps 70 \
--fps 24 \
--output_dir cat_1

# TI2I ロボット1_0
# https://www.cicasoft.com.tr/blog-detay-279542-Yapay-Zeka-nin-Egitim-Sistemi-Icindeki-Onemi.html
python3 inference/batch_inference.py \
--model_type video2world \
--prompt "A friendly humanoid robot stands on a futuristic city street at dusk. Neon signs reflect off its smooth metallic body. The robot slowly raises its right arm, waves gently, then tilts its head slightly as if recognizing someone. Passing cars blur in the background while soft ambient lights flicker. The camera slightly dolly-zooms forward, emphasizing depth and subtle reflections." \
--image_path "images/robot/robot1.jpg" \
--num_seeds 10 \
--num_steps 70 \
--fps 24 \
--guidance 7.5 \
--output_dir robot1_0

# TI2I ロボット1_1
# https://www.cicasoft.com.tr/blog-detay-279542-Yapay-Zeka-nin-Egitim-Sistemi-Icindeki-Onemi.html
python3 inference/batch_inference.py \
--model_type video2world \
--prompt "The small robot subtly shifts its weight from one foot to another, maintaining balance while its head slowly moves side to side as if scanning the surroundings. Its glossy surface reflects the colorful city lights dynamically. Small mechanical servos softly hum as the movements transition smoothly. The camera gently zooms in to emphasize the balance corrections." \
--image_path "images/robot/robot1.jpg" \
--num_seeds 10 \
--num_steps 70 \
--fps 24 \
--guidance 7.5 \
--output_dir robot1_1
