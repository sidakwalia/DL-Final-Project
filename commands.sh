singularity shell --nv --overlay /scratch/mm12318/DL/flickr8k.sqf docker://mehtamohit013/nerf_cuda:11.8.0-devel-cudnn8-ubuntu22.04-all
sbatch --cpus-per-task=2 --time=01:00:00 --mem=1GB --wrap "sleep infinity"
sbatch --cpus-per-task=8 --time=04:00:00 --mem=16GB --gres=gpu:1 --wrap "sleep infinity"


torchrun --nproc_per_node=1 run_beit3_finetuning.py --model beit3_base_patch16_480 --input_size 480 --task coco_captioning --batch_size 1 --layer_decay 1.0 --lr 4e-5 --randaug --epochs 10 --warmup_epochs 1 --drop_path 0.1 --sentencepiece_model /home/mm12318/DL_Class/BEiT/beit3.spm --finetune /home/mm12318/DL_Class/BEiT/beit3_base_patch16_224.pth --data_path /COCO --output_dir ./output_dir --log_dir ./log_dir --weight_decay 0.05 --seed 42 --save_ckpt_freq 5 --num_max_bpe_tokens 32 --captioning_mask_prob 0.7 --drop_worst_after 12000 --dist_eval --checkpoint_activations --enable_deepspeed
