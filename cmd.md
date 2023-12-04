# Finetune with LoRA
deepspeed llava/train/train_xformers.py \
    --lora_enable True --lora_r 16 --lora_alpha 32 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /mnt/workspace/llava-v1.5-7b \
    --version v1 \
    --data_path /mnt/workspace/minigpt4_data.json \
    --image_folder '' \
    --vision_tower /mnt/workspace/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --fp16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-minigpt4-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

# config
r=16, alpha=32, bs=1, accumu=4, GPU=16.8G, 1 epoch, time=1 hour
r=32, alpha=64, bs=1, accumu=4, GPU=18.0G, 1 epoch, time=1 hour
r=64, alpha=128, bs=1, accumu=4, GPU=20.9G, 1 epoch, time=1 hour
