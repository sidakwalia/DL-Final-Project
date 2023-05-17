# Fine Tuning Image Transformers

We have fine-tuned our model on the COCO dataset for image captioning and Visual Question Answering. We have used the BeiT-3 model for visual transformers.

There are two different methods used here:

1. Image Captioning
2. Visual Question Answering

## Fine-tuning BEiT-3 on Image Captioning 
Command used to run the model: 

bash
torchrun --nproc_per_node=1 run_beit3_finetuning.py --model beit3_base_patch16_480 --input_size 480 --task coco_captioning --batch_size 32 --layer_decay 1.0 --lr 4e-5 --randaug --epochs 10 --warmup_epochs 1 --drop_path 0.1 --sentencepiece_model beit3.spm --finetune beit3_base_patch16_224.pth --data_path /COCO --output_dir ./output_freeze --log_dir ./log_freeze --weight_decay 0.05 --seed 42 --save_ckpt_freq 1 --num_max_bpe_tokens 32 --captioning_mask_prob 0.7 --drop_worst_after 12000 --dist_eval  --enable_deepspeed > log_freeze.txt 2>&1

- Pretrained model weights COCO captioning dataset can be found [here]https://drive.google.com/drive/u/0/folders/1q8Z2HDEZvCxqPvRuJCBJdXblgOXIWDZK


## Fine-tuning BEiT-3 on VQAv2 (Visual Question Answering)

### Setup

1. [Setup environment](../README.md#setup).
2. Download COCO:
   - [2014 train images](http://images.cocodataset.org/zips/train2014.zip)
   - [2014 val images](http://images.cocodataset.org/zips/val2014.zip)
   - [2015 test images](http://images.cocodataset.org/zips/test2015.zip)
   - Annotations: [train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip), [val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip)
   - Questions: [train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip), [val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip), [test](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip)

Organize the dataset as following:


```
/path/to/your_data/
  train2014/            
    COCO_train2014_000000000009.jpg                
    ...
  val2014/              
    COCO_val2014_000000000042.jpg
    ...  
  test2015/              
    COCO_test2015_000000000001.jpg
    ...         
  vqa/
    v2_OpenEnded_mscoco_train2014_questions.json
    v2_OpenEnded_mscoco_val2014_questions.json
    v2_OpenEnded_mscoco_test2015_questions.json
    v2_OpenEnded_mscoco_test-dev2015_questions.json
    v2_mscoco_train2014_annotations.json
    v2_mscoco_val2014_annotations.json
```


Generate the index JSON files using the following command. [beit3.spm](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/sentencepiece/beit3.spm) is the sentencepiece model used for tokenizing texts.

```python
from datasets import VQAv2Dataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/your_beit3_model_path/beit3.spm")

VQAv2Dataset.make_dataset_index(
    data_path="/path/to/your_data",
    tokenizer=tokenizer,
    annotation_data_path="/path/to/your_data/vqa",
)
```bash


To fine-tune the model on the COCO dataset, run the following command:

```bash
bash bash.sh

This command consist of this below :


python3 run_beit3_finetuning.py \
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
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_base_patch16_224.pth \
        --data_path /path/to/your_data \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model/log \
        --weight_decay 0.01 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --task_head_lr_weight 20 \
        --opt_betas 0.9 0.98 \
        --enable_deepspeed

Once the model has been fine-tuned, you can make predictions on the COCO dataset by running:

Here is an explanation of the parameters used in the command:

- `--model beit3_base_patch16_480`: Specifies the BEiT model to be used, in this case, the base model with a patch size of 16 and input size of 480.

- `--input_size 480`: Defines the input size for the model.

- `--task vqav2`: Specifies the task to be performed, here it is Visual Question Answering version 2.

- `--batch_size 16`: Sets the batch size to 16. The batch size is the number of training examples used in one iteration.

- `--layer_decay 1.0`: The rate at which the layer learning rates decay.

- `--lr 3e-5`: Sets the learning rate for the model.

- `--update_freq 1`: Update frequency for the model weights.

- `--randaug`: Enables RandAugment, a data augmentation method for automatic search of data augmentation policies.

- `--epochs 10`: The number of complete passes over the dataset during training.

- `--warmup_epochs 1`: The number of warmup epochs for learning rate scheduler.

- `--drop_path 0.1`: The drop path rate for the stochastic depth.

- `--sentencepiece_model /your_beit3_model_path/beit3.spm`: Path to the SentencePiece model used for tokenization.

- `--finetune /your_beit3_model_path/beit3_base_patch16_224.pth`: The model path for fine-tuning.

- `--data_path /path/to/your_data`: The path to the directory where the data is stored.

- `--output_dir /path/to/save/your_model`: Directory where the output model will be saved.

- `--log_dir /path/to/save/your_model/log`: Directory where the logs will be stored.

- `--weight_decay 0.01`: The weight decay for the optimizer.

- `--seed 42`: Sets the random seed for reproducibility.

- `--save_ckpt_freq 5`: The frequency at which the model checkpoints are saved.

- `--task_head_lr_weight 20`: The learning rate weight for the task-specific head.

- `--opt_betas 0.9 0.98`: The beta parameters for the Adam optimizer.

- `--enable_deepspeed`: Enable DeepSpeed for training acceleration.


bash prediction.sh

Here is an explanation of the parameters used in the command:

- `--model beit3_base_patch16_480`: This specifies the model architecture to use. In this case, the BEiT model with base configuration, patch size of 16 and input size of 480 is used.

- `--input_size 480`: This defines the size of the input images that the model should expect.

- `--task vqav2`: This specifies the task to be performed, which is Visual Question Answering version 2 (VQAv2) in this case.

- `--batch_size 64`: This sets the batch size to 64. The batch size is the number of samples that will be passed through the network at once.

- `--sentencepiece_model beit3.spm`: This is the path to the SentencePiece model used for tokenization.

- `--finetune beit3_base_patch16_480_vqa.pth`: This is the path to the pre-trained model that will be fine-tuned on the new task.

- `--data_path dataset`: This is the path to the directory where the data for the task is stored.

- `--output_dir your_prediction`: This is the directory where the model's output will be saved.

- `--eval`: This flag indicates that the model should be evaluated after training.

- `--dist_eval`: This flag is used to enable distributed evaluation if multiple GPUs are available.
