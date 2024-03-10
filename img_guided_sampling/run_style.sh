
python style_transfer.py \
    --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
    --prompt "a cat wearing glasses." \
    --style_ref_img_path "./assets/style_imgs/jojo.jpeg" \
    --outdir ./outputs/style/ \
    --num_inference_steps 100 \
    --num_pred_steps 4 \
    --pred_type 'type-2' \
    --seed 2023