import pandas as pd
import numpy as np
import os
import json
import requests
from tqdm import tqdm

def get_data(img, save_path, desc="Progress"):
    os.makedirs(save_path, exist_ok=True)
    for i in tqdm(range(len(img)), desc=desc):
        url = img[i]['coco_url']
        filename = img[i]['file_name']

        # Get the image from the URL
        img_data = requests.get(url).content
        # save the image data to the file
        with open(save_path + "/" + filename, 'wb') as handler:
            handler.write(img_data)

data_pth = "nocaps_val_4500_captions.json"
f = open(data_pth,'r+')
val_img = json.load(f)['images']
f.close()

data_pth = "nocaps_test_image_info.json"
f = open(data_pth,'r+')
test_img = json.load(f)['images']
f.close()

get_data(val_img, "val", desc="Downloading Val Images")
get_data(test_img, "test", desc="Downloading Test Images")