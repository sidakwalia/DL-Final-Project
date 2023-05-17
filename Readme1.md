# Finetuning Image Transformers


# Project Structure
| Directory / File | Description |
|-----------------|-----------------|
| COCO_Image_captioning | Contains Beit-3 model trained on COCO captioning images | 
| COCO_VQAv2 | Contains Beit-3 model trained on VQAv2 images | 

How to run Beit-3 for COCO dataset -

torchrun --nproc_per_node=1 run_beit3_finetuning.py --model beit3_base_patch16_480 --input_size 480 --task coco_captioning --batch_size 32 --layer_decay 1.0 --lr 4e-5 --randaug --epochs 10 --warmup_epochs 1 --drop_path 0.1 --sentencepiece_model /DL_Class/BEiT/beit3.spm --finetune /DL_Class/BEiT/beit3_base_patch16_224.pth --data_path /COCO --output_dir ./output_freeze --log_dir ./log_freeze --weight_decay 0.05 --seed 42 --save_ckpt_freq 1 --num_max_bpe_tokens 32 --captioning_mask_prob 0.7 --drop_worst_after 12000 --dist_eval  --enable_deepspeed > log_freeze.txt 2>&1


Weights for COCO captioning dataset can be found here -
https://drive.google.com/drive/u/1/folders/1q8Z2HDEZvCxqPvRuJCBJdXblgOXIWDZK

COCO captioning: Tensorboard logs https://github.com/sidakwalia/DL-Final-Project/tree/main/COCO_Image_captioning/log_freeze

COCO captioning log file: https://github.com/sidakwalia/DL-Final-Project/blob/main/COCO_Image_captioning/log_freeze.txt


# Model Architecture

The Beit-3 model architecture used in this project is based on this paper -
Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu
Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, Furu Wei

https://arxiv.org/pdf/2208.10442.pdf


# References

PyTorch documentation
