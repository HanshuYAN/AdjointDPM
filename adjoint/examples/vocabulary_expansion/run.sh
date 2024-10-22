python voca_expansion.py \
    --output_dir './voca_expansion_dog/' \
    --pretrained_fgvc_path "./assets/dog.pth.tar" \
    --prompt 'A Cairn, a type of dog' \
    --model_guidance_dataset 'dog' \
    --model_guidance_class 41 \
    --grad_scale 0.05 \
    --num_train_epochs 30 \
    --learning_rate 0.01 