import requests
import zipfile
import os

def download_and_unzip(url, extract_to='.'):
    """
    Download a ZIP file and extract its contents in memory
    yields (filename, file-like object) pairs
    """
    response = requests.get(url)
    with open('file.zip', 'wb') as f:
        f.write(response.content)

    with zipfile.ZipFile('file.zip', 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Use the function
download_and_unzip('http://images.cocodataset.org/zips/train2014.zip','/home/paperspace/.gnupg/project/DL-Final-Project/unilm/beit3/dataset/')
os.remove('file.zip')
download_and_unzip('http://images.cocodataset.org/zips/val2014.zip','/home/paperspace/.gnupg/project/DL-Final-Project/unilm/beit3/dataset/')
print("ended here line 20")
os.remove('file.zip')
download_and_unzip('http://images.cocodataset.org/zips/test2015.zip','/home/paperspace/.gnupg/project/DL-Final-Project/unilm/beit3/dataset/')
print("ended here line 22")
os.remove('file.zip')
# download_and_unzip('https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip','/home/paperspace/.gnupg/project/DL-Final-Project/unilm/beit3/dataset/vqa')
# print("ended here line 25")

download_and_unzip('https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip','/home/paperspace/.gnupg/project/DL-Final-Project/unilm/beit3/dataset/vqa')
download_and_unzip('https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip','/home/paperspace/.gnupg/project/DL-Final-Project/unilm/beit3/dataset/vqa')
download_and_unzip('https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip','/home/paperspace/.gnupg/project/DL-Final-Project/unilm/beit3/dataset/vqa')
download_and_unzip('https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip','/home/paperspace/.gnupg/project/DL-Final-Project/unilm/beit3/dataset/vqa')
