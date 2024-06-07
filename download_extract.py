#download_extract.py
import os
import wget
import zipfile

def download_and_extract():
    if not os.path.exists("data/Flicker8k_Dataset"):
        os.makedirs("data", exist_ok=True)
        wget.download("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip", out="data")
        with zipfile.ZipFile("data/Flickr8k_Dataset.zip", "r") as zip_ref:
            zip_ref.extractall("data")
        os.remove("data/Flickr8k_Dataset.zip")
        
    if not os.path.exists("data/Flickr8k.token.txt"):
        os.makedirs("data", exist_ok=True)
        wget.download("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip", out="data")
        with zipfile.ZipFile("data/Flickr8k_text.zip", "r") as zip_ref:
            zip_ref.extractall("data")
        os.remove("data/Flickr8k_text.zip")
        os.remove("Flickr8k_text.zip")

if __name__ == "__main__":
    download_and_extract()