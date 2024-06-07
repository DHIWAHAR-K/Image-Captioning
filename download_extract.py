#download_extract.py
import os
import wget
import zipfile

def download_and_extract():
    if not os.path.exists("Flicker8k_Dataset"):
        wget.download("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip")
        with zipfile.ZipFile("Flickr8k_Dataset.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove("Flickr8k_Dataset.zip")
        
    if not os.path.exists("Flickr8k.token.txt"):
        wget.download("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip")
        with zipfile.ZipFile("Flickr8k_text.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove("Flickr8k_text.zip")

if __name__ == "__main__":
    download_and_extract()