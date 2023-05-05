singularity shell --nv --overlay /scratch/mm12318/DL/flickr8k.sqf docker://mehtamohit013/nerf_cuda:11.8.0-devel-cudnn8-ubuntu22.04-all
sbatch --cpus-per-task=2 --time=01:00:00 --mem=1GB --wrap "sleep infinity"
sbatch --cpus-per-task=8 --time=04:00:00 --mem=16GB --gres=gpu:1 --wrap "sleep infinity"