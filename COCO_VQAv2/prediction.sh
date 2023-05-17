python run_beit3_finetuning.py \
        --model beit3_base_patch16_480 \
        --input_size 480 \
        --task vqav2 \
        --batch_size 64 \
        --sentencepiece_model beit3.spm \
        --finetune beit3_base_patch16_480_vqa.pth \
        --data_path dataset \
        --output_dir your_prediction \
        --eval \
        --dist_eval