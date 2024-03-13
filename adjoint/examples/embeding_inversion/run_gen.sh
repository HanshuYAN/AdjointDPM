### An example to generate data for embedding inversion
python3 synthetic_data_gen.py \
    --initial_prompt 'Photo from a city street in the 1970s' \
    --after_prompt 'Photo from a city street in the 1970s in a style of Van Gogh' \
    --save_path './assets/TrainPrompt' \
    --index 0