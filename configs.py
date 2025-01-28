# #configs.py
# EPOCHS = 5
# UNITS = 512
# MAX_LENGTH = 40
# BATCH_SIZE = 64
# BUFFER_SIZE = 1000
# EMBEDDING_DIM = 512
# VOCABULARY_SIZE = 15000


# data_path = "./data"

import os
from dotenv import load_dotenv
from utils import ensure_directories_exist

# Load environment variables
load_dotenv()

# General Configurations
DATA_PATH = os.getenv("DATA_PATH", "./data")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/model.h5")
VOCABULARY_PATH = os.getenv("VOCABULARY_PATH", "./vocabulary/vocab.pkl")
LOGS_PATH = os.getenv("LOGS_PATH", "./logs")

# Training Configurations
EPOCHS = int(os.getenv("EPOCHS", 5))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", 1000))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 512))
UNITS = int(os.getenv("UNITS", 512))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 40))
VOCABULARY_SIZE = int(os.getenv("VOCABULARY_SIZE", 15000))

# Ensure required directories exist
required_directories = [
    os.path.dirname(MODEL_PATH),
    os.path.dirname(VOCABULARY_PATH),
    LOGS_PATH,
]
ensure_directories_exist(required_directories)