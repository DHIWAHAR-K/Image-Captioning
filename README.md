# Image Captioning

This project implements an **end-to-end image captioning system** using a **CNN for feature extraction** and a **Transformer encoder-decoder architecture** for sequence generation. It generates captions for images from the MS COCO dataset.

---

## 📁 Project Structure

| File/Module              | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `configs.py`             | Stores model/data/training configurations. Loads from `.env`.               |
| `data_cleaning.py`       | Preprocesses MS COCO captions and links images with captions.               |
| `data_generation.py`     | Splits the data into train/validation and generates image-caption pairs.    |
| `data_augmentation.py`   | Builds `tf.data.Dataset` pipelines and applies real-time image augmentation.|
| `tokenizer.py`           | Initializes and fits a `TextVectorization` tokenizer on cleaned captions.   |
| `vocabulary.py`          | Generates `word2idx` and `idx2word` mappings and caption dictionary.        |
| `feature_extraction.py`  | Extracts image embeddings using InceptionV3.                                |
| `embeddings.py`          | Defines token and positional embeddings.                                    |
| `transformer_encoder.py` | Implements the Transformer encoder layer.                                   |
| `transformer_decoder.py` | Implements the Transformer decoder layer.                                   |
| `model.py`               | Combines encoder, decoder, and CNN into one model. Handles training logic.  |
| `train.py`               | Main script to compile, train, evaluate, and save the captioning model.     |
| `utils.py`               | Utility function to create directories if they don’t exist.                 |

---

## ⚙️ Configuration Parameters

| Parameter         | Description                                        |
|-------------------|----------------------------------------------------|
| `DATA_PATH`       | Path to dataset directory (`./data`)               |
| `MODEL_PATH`      | Path to save trained model (`./models/model.h5`)   |
| `VOCABULARY_PATH` | Path to save vocabulary file (`./vocabulary/vocab.pkl`) |
| `LOGS_PATH`       | Directory to store logs                            |
| `EPOCHS`          | Number of training epochs (`5`)                    |
| `BATCH_SIZE`      | Training batch size (`64`)                         |
| `BUFFER_SIZE`     | Buffer size for shuffling data (`1000`)            |
| `EMBEDDING_DIM`   | Token/position embedding dimension (`512`)         |
| `UNITS`           | Hidden units in transformer layers (`512`)         |
| `MAX_LENGTH`      | Max sequence length for captions (`40`)            |
| `VOCABULARY_SIZE` | Vocabulary size used by tokenizer (`15000`)        |

---

## Model Architecture

### CNN Encoder
- Based on InceptionV3 (pretrained on ImageNet)
- Removes top layers and reshapes output

### Transformer Encoder
- Single-layer MHA + Feedforward Network

### Transformer Decoder
- Token + Position Embeddings
- Multi-head attention over:
  - Caption tokens (self-attention)
  - CNN-encoded image features
- Output: Sequence of token predictions

---

## Setup

### 1. Install dependencies

```bash
pip install tensorflow pandas matplotlib scikit-learn python-dotenv
```

### 2. Download the MS COCO dataset
	- Download captions: captions_train2017.json
	-	Download images: train2017.zip
	-	Extract them into ./data/train2017/ and ./data/annotations/

 ## Training the model
 To train the captioning model, run:
 ```bash
 python train.py
 ```

This will:
	•	Load the dataset and preprocess captions
	•	Tokenize captions using a custom vocabulary
	•	Extract features using InceptionV3
	•	Train the model using a CNN + Transformer architecture
	•	Save the trained model to model.h5
	•	Plot training and validation loss curves


 ## Output
	- ./models/model.h5 – Trained image captioning model
	- ./vocabulary/vocab.pkl – Vocabulary used by the tokenizer
	- Training/validation loss curves

 ## Features
	- End-to-end trainable image captioning pipeline
	- Uses InceptionV3 for visual features
	- Trains Transformer from scratch (Encoder + Decoder)
	- Token-level positional embeddings
	- Custom caption tokenizer with padding and start/end tokens
	- Real-time image augmentation (flip, contrast, rotation)

## License

This project is for research and educational purposes. Feel free to reuse and extend with attribution.
