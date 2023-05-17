# Fine Tuning Image Transformers

We have fine-tuned our model on the COCO dataset for image captioning and Visual Question Answering. We have used the BeiT-3 model for visual transformers.

There are two different methods used here:

1. Image Captioning
2. Visual Question Answering

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
python3 run_beit3_finetuning.py [For finetuning]

Runbash prediction.sh for making predictions on coco-dataset.
