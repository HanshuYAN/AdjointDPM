#!/bin/bash
DEVICES='0,'

#########################

# ImageNet128 with classifier guidance (large guidance scale) example to generate adversarial samples
# Set the fixed_class to be the class you want to attack

data="imagenet128_guided"
scale="8.0"
sampleMethod='neuralode'
steps="30"
DIS="time_uniform"
fixed_class=42
workdir="./experiments/"$data"/"$sampleMethod"_"$steps"_class"$fixed_class

python main.py --config $data".yml" --exp=$workdir --attack --fixed_class=$fixed_class --timesteps=$steps --eta 0 --ni --skip_type=$DIS --sample_type=$sampleMethod --port 12350 --scale=$scale --num_samples 30 --attack_batch_size 10 --nb_iter 30 --lr 0.01 --clamp_min 0.1 --clamp_max 0.1
