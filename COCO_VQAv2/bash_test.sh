python3  run_beit3_finetuning.py \
        --model beit3_base_patch16_480 \
        --input_size 480 \
        --task vqav2 \
        --batch_size 16 \
        --layer_decay 1.0 \
        --lr 3e-5 \
        --update_freq 1 \
        --randaug \
        --epochs 10 \
        --warmup_epochs 1 \
        --drop_path 0.1 \
        --sentencepiece_model beit3.spm \
        --finetune beit3_base_patch16_480_vqa.pth \
        --data_path dataset \
        --output_dir model \
        --log_dir log \
        --weight_decay 0.01 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --task_head_lr_weight 20 \
        --opt_betas 0.9 0.98 \
        --enable_deepspeed